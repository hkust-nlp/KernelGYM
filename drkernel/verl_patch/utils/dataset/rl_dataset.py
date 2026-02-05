# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.model import compute_position_id_with_mask


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO

    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key='prompt',
        image_key='images',
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir='~/.cache/verl/rlhf',
        chat_template_func=None,
        apply_chat_template=False,
        return_raw_chat=False,
        truncation='error',
        # if using sample_size, trucnate every validation dataset into this to reduce the time used for evaluation each time
        sample_size=None,
        filter_overlong_prompts=True,
        system_prompt_config=None,
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.apply_chat_template = apply_chat_template
        self.truncation = truncation
        self.sample_size = sample_size
        self.filter_overlong_prompts = filter_overlong_prompts
        self.system_prompt_config = system_prompt_config

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            if self.sample_size is not None and len(dataframe) > self.sample_size:
                # use random state to ensure it can be reproducible
                dataframe = dataframe.sample(n=self.sample_size, random_state=42)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        if self.filter_overlong_prompts:
            # filter out too long prompts
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            if self.apply_chat_template:
                self.dataframe = self.dataframe[
                    self.dataframe.apply(
                        lambda doc: len(
                            tokenizer.encode(
                                tokenizer.apply_chat_template(
                                    doc[prompt_key], add_generation_prompt=True, tokenize=False
                                )
                            )
                        )
                        <= self.max_prompt_length,
                        axis=1,
                    )
                ]
            else:
                self.dataframe = self.dataframe[
                    self.dataframe.apply(
                        lambda doc: len(tokenizer.encode(doc[prompt_key][0]['content'])) <= self.max_prompt_length,
                        axis=1,
                    )
                ]

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = copy.deepcopy(row_dict.pop(self.prompt_key))

        # Apply prompt from config file if provided
        if self.system_prompt_config is not None and self.apply_chat_template is True:
            # Load prompt config from file
            if os.path.exists(self.system_prompt_config):
                try:
                    with open(self.system_prompt_config) as f:
                        content = f.read().strip()

                    # Parse based on file extension
                    file_ext = os.path.splitext(self.system_prompt_config)[1].lower()
                    if file_ext == '.json':
                        config = json.loads(content)
                        prompt_text = config.get('prompt', '')
                        prompt_config_method = config.get('method', 'system')
                    elif file_ext in ['.yaml', '.yml']:
                        config = OmegaConf.load(self.system_prompt_config)
                        prompt_text = config.get('prompt', '')
                        prompt_config_method = config.get('method', 'system')
                    else:
                        # Treat as plain text
                        prompt_text = content
                        prompt_config_method = 'system'

                    # Apply prompt based on method
                    if prompt_config_method == 'system':
                        # Add system message to the beginning of the chat if not already present
                        if prompt_text and (not chat or chat[0].get('role') != 'system'):
                            chat = [{'role': 'system', 'content': prompt_text}] + chat
                    elif prompt_config_method == 'pre_input':
                        # Insert before the first user input
                        for i, message in enumerate(chat):
                            if message.get('role') == 'user':
                                new_content = prompt_text + message.get('content', '')
                                chat[i]['content'] = new_content
                                break
                    elif prompt_config_method == 'post_input':
                        # Insert after the first user input
                        for i, message in enumerate(chat):
                            if message.get('role') == 'user':
                                new_content = message.get('content', '') + prompt_text
                                chat[i]['content'] = new_content
                                break
                except Exception as e:
                    print(f"Error processing prompt config file: {e}")
            else:
                print(f"Prompt config file {self.system_prompt_config} does not exist.")
        elif self.system_prompt_config is not None and self.apply_chat_template is False:
            raise ValueError("Error: system_prompt_config is provided but apply_chat_template is False.")

        if self.apply_chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt_with_chat_template = chat[0]['content']

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>'
                        + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length)
                        + '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace(
                    '<|placeholder|>', self.processor.image_token
                )
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist() if not isinstance(chat, list) else chat

        # add index for each prompt
        extra_info = row_dict.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        index = extra_info.get("index", 0)
        row_dict["index"] = index
        row_dict["prompt_index"] = item

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()


class SolveRateDynamicRLHFDataset(RLHFDataset):
    """
    RLHF Dataset with dynamic solve rate tracking capabilities

    Attributes:
        current_solve_rates (np.ndarray): Array of current solve rates for samples
        original_indices (list): Preserved original indices for data access

    Args:
        Inherits all arguments from RLHFDataset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Validate dataset structure
        if 'solve_rate' not in self.dataframe.columns:
            raise ValueError("Dataset must contain 'solve_rate' column")

        # Initialize solve rate tracking
        self.current_solve_rates = self.dataframe['solve_rate'].to_numpy().copy()
        # self.original_indices = self.dataframe.index.tolist()

    def __len__(self):
        return super().__len__()
