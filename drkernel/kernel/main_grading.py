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
"""
Kernel grading script for evaluating model-generated kernel code.
Adapted from verl_patch/trainer/code/main_grading.py for kernel-specific evaluation.
"""
from audioop import mul
import json
import os
import time
import re
import uuid
from uuid import uuid4
from collections import defaultdict
from functools import partial
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import hydra
import numpy as np
import pandas as pd
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs

from verl_patch.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl_patch.workers.code.fsdp_workers import ActorRolloutRefWorker
from verl_patch.trainer.code.metrics.multi_turn_metrics import compute_multi_turn_metrics
from kernel.metrics.kernel_multi_turn_metrics import compute_kernel_multi_turn_metrics
from verl_patch.utils.tracking import ValidationGenerationsLogger
from .constant import QWEN3CHATTEMPLATE

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)

def _parse_turn_id_from_filename(filename: str) -> int:
    """Extract numeric turn_id from a filename like turn_12_eval.json."""
    match = re.search(r"turn_(\d+)_eval\.json$", filename)
    if match:
        return int(match.group(1))
    return int(1e9)

def _parse_problem_sample_dir(name: str) -> tuple:
    """Extract (problem_id, sample_id) from problem_<pid>_sample_<sid>."""
    match = re.match(r"problem_(\d+)_sample_(\d+)$", name)
    if match:
        return (int(match.group(1)), int(match.group(2)), name)
    return (int(1e9), int(1e9), name)

def compute_pass_at_k(results: np.ndarray, k: int, threshold: float = 0.99) -> float:
    """
    Compute the average pass@k metric for a list of problem results.

    Args:
        results: A numpy array of shape (num_problems, num_samples) containing scores.
        k: The number of samples to consider (k in pass@k).
        threshold: The threshold above which a sample is considered as passing.

    Returns:
        The average pass@k score across all problems.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    num_problems, num_samples = results.shape
    if num_samples < k:
        raise ValueError(f"Each problem must have at least {k} samples, found {num_samples}")

    pass_rates = []
    for problem_scores in results:
        # Convert scores to binary pass/fail based on threshold
        passes = (problem_scores >= threshold).astype(int)
        # If any of the k samples pass, the problem is considered solved
        # Standard pass@k: probability that at least one of k samples passes
        num_passes = passes.sum()
        if num_passes >= k:
            # At least k samples passed, so pass@k = 1.0
            pass_rate = 1.0
        elif num_passes == 0:
            # No samples passed, so pass@k = 0.0
            pass_rate = 0.0
        else:
            # Some samples passed (but < k)
            # Use combinatorial formula: 1 - C(n-c, k) / C(n, k)
            # where n = total samples, c = number of passes
            from math import comb
            n = num_samples
            c = num_passes
            if c >= k:
                pass_rate = 1.0
            else:
                pass_rate = 1.0 - (comb(n - c, k) / comb(n, k))

        pass_rates.append(pass_rate)

    return float(np.mean(pass_rates))


def get_custom_reward_fn(config):
    import importlib.util
    import os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


def get_reward_manager_cls(config):
    """Get reward manager class based on config, supporting kernel-specific managers."""
    reward_manager_name = (
        config.reward_model.get("reward_manager")
        if hasattr(config, "reward_model")
        else None
    )

    if reward_manager_name == 'naive':
        from verl_patch.workers.code.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'code':
        from verl_patch.workers.code.reward_manager import CodeRewardManager
        reward_manager_cls = CodeRewardManager
    elif reward_manager_name == 'sandbox':
        from verl_patch.workers.code.reward_manager import HttpSandboxRewardManager
        reward_manager_cls = HttpSandboxRewardManager
    elif reward_manager_name == 'math':
        from verl_patch.workers.code.reward_manager import MathRewardManager
        reward_manager_cls = MathRewardManager
    elif reward_manager_name == 'swe':
        from verl_patch.workers.code.reward_manager import SWERewardManager
        reward_manager_cls = SWERewardManager
    elif reward_manager_name == 'kernel':
        from verl_patch.workers.code.reward_manager import KernelRewardManager
        reward_manager_cls = KernelRewardManager
    elif reward_manager_name == 'kernel_async':
        from kernel.workers.reward_manager.kernel_async import AsyncKernelRewardManager
        reward_manager_cls = AsyncKernelRewardManager
    else:
        raise NotImplementedError(f"Reward manager '{reward_manager_name}' not implemented")

    return reward_manager_cls


def _check_kernel_server_health(server_url: str) -> None:
    """
    Check KernelServer health status
    """
    import asyncio
    import httpx
    import logging

    logger = logging.getLogger(__name__)

    async def check_health():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                health_url = f"{server_url}/health"
                logger.info(f"Checking KernelServer health at: {health_url}")
                response = await client.get(health_url)
                response.raise_for_status()
                health_data = response.json()
                logger.info(f"KernelServer health response: {health_data}")
                if health_data.get("status") != "healthy":
                    raise RuntimeError(f"KernelServer is not healthy: {health_data}")
                logger.info("✅ KernelServer health check passed")
                return True
        except Exception as e:
            logger.error(f"❌ KernelServer health check failed: {e}")
            raise RuntimeError(f"KernelServer at {server_url} is not accessible: {e}")

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(check_health())


# ============================================================================
# Gradio Visualization - Modular and Extensible Design
# ============================================================================

# --- Utility Functions ---

def safe_load_file(file_path, default="# File not found"):
    """Safely load file with error handling."""
    try:
        if not os.path.exists(file_path):
            return default
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content if content.strip() else default
    except Exception as e:
        return f"# Error loading file: {e}"


def truncate_large_text(text, max_chars=100000):
    """Truncate large text files."""
    if len(text) <= max_chars:
        return text
    # Keep first and last portions
    half = max_chars // 2
    return text[:half] + f"\n\n... [truncated {len(text) - max_chars} characters] ...\n\n" + text[-half:]


def format_boolean(value):
    """Format boolean for display: True → '✓', False → '✗'"""
    if isinstance(value, bool):
        return '✓' if value else '✗'
    return str(value)


def format_percentage(value):
    """Format float as percentage: 0.452 → '45.2%'"""
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return str(value)


# --- Configuration ---

@dataclass
class VisualizerConfig:
    """Centralized configuration for visualization."""
    sidebar_scale: int = 2  # 加宽 sidebar：从 1 改为 2
    main_area_scale: int = 3
    conversation_max_lines: int = 50
    max_tabs: int = 10
    error_truncate_chars: int = 50
    file_size_limit_bytes: int = 100 * 1024  # 100KB


# --- Data Loader Classes ---

class EvalDataLoader(ABC):
    """Abstract base class for loading evaluation data."""

    @abstractmethod
    def load_samples(self, eval_dir):
        """Returns list of sample dicts with standardized structure."""
        raise NotImplementedError

    @abstractmethod
    def is_applicable(self, eval_dir):
        """Check if this loader can handle the directory structure."""
        raise NotImplementedError


class SingleTurnLoader(EvalDataLoader):
    """Loads flat file structure: problem_X_sample_Y_*.json/py"""

    def is_applicable(self, eval_dir):
        """Check if this is a single-turn flat structure."""
        import glob
        eval_files = glob.glob(os.path.join(eval_dir, "*_eval.json"))
        # Single-turn if we have flat _eval.json files
        return len(eval_files) > 0

    def load_samples(self, eval_dir):
        """Load single-turn samples from flat file structure."""
        import glob
        samples = []
        files = glob.glob(os.path.join(eval_dir, "*_eval.json"))

        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    data['_filename'] = f
                    data['is_multi_turn'] = False

                    # Get code filename
                    base_name = f.replace('_eval.json', '')
                    code_file = base_name + '_kernel.py'
                    data['_code'] = safe_load_file(code_file, "# Code file not found")

                    # Get reference code filename
                    ref_file = base_name + '_ref.py'
                    ref_content = safe_load_file(ref_file, "# Reference code not found")
                    data['_ref'] = ref_content

                    samples.append(data)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        # Sort samples by Problem ID then Sample ID
        def sort_key(s):
            try:
                pid = str(s.get('problem_id', '0'))
                sid = str(s.get('sample_id', '0'))
                if pid.isdigit() and sid.isdigit():
                    return (int(pid), int(sid))
                return (pid, sid)
            except:
                return (0, 0)

        samples.sort(key=sort_key)
        return samples


class MultiTurnKernelLoader(EvalDataLoader):
    """Loads kernel multi-turn structure: problem_X_sample_Y/turn_*"""

    def is_applicable(self, eval_dir):
        """Check if this is a multi-turn directory structure."""
        conv_dirs = [d for d in os.listdir(eval_dir)
                     if os.path.isdir(os.path.join(eval_dir, d)) and d.startswith("problem_")]
        return len(conv_dirs) > 0

    def load_samples(self, eval_dir):
        """Load multi-turn samples from conversation directories."""
        import glob
        samples = []
        conv_dirs = [d for d in os.listdir(eval_dir)
                     if os.path.isdir(os.path.join(eval_dir, d)) and d.startswith("problem_")]

        for conv_dir_name in sorted(conv_dirs, key=_parse_problem_sample_dir):
            conv_dir = os.path.join(eval_dir, conv_dir_name)

            # Load all turn eval files
            turn_files = sorted(
                [f for f in os.listdir(conv_dir) if f.startswith("turn_") and f.endswith("_eval.json")],
                key=_parse_turn_id_from_filename,
            )

            if not turn_files:
                continue

            # Extract problem_id and sample_id from directory name
            # Format: problem_<pid>_sample_<sid>
            try:
                parts = conv_dir_name.split('_')
                problem_id = int(parts[1])
                sample_id = int(parts[3])
            except:
                problem_id = 0
                sample_id = 0

            # Load turns data
            turns = []
            total_score = 0.0
            for turn_file in turn_files:
                turn_path = os.path.join(conv_dir, turn_file)
                try:
                    with open(turn_path, 'r') as f:
                        turn_eval = json.load(f)

                    turn_id = turn_eval.get('turn_id', 0)

                    # Load corresponding kernel code
                    kernel_file = os.path.join(conv_dir, f"turn_{turn_id}_kernel.py")
                    kernel_code = safe_load_file(kernel_file, f"# Code not found for turn {turn_id}")

                    turns.append({
                        'turn_id': turn_id,
                        'code': kernel_code,
                        'eval': turn_eval,
                    })

                    total_score += turn_eval.get('score', 0.0)
                except Exception as e:
                    print(f"Error loading {turn_path}: {e}")
            # Ensure turns are ordered by numeric turn_id
            turns.sort(key=lambda t: int(t.get('turn_id', 0)))

            # Load shared files
            reference_code = safe_load_file(
                os.path.join(conv_dir, "reference.py"),
                "# Reference code not found"
            )

            conversation_text = safe_load_file(
                os.path.join(conv_dir, "full_conversation.txt"),
                "Conversation history not available"
            )
            conversation_text = truncate_large_text(conversation_text)

            # Create sample dict
            sample = {
                'problem_id': problem_id,
                'sample_id': sample_id,
                'is_multi_turn': True,
                'conversation_text': conversation_text,
                'reference_code': reference_code,
                'turns': turns,
                'total_score': total_score,
                'num_turns': len(turns),
            }

            samples.append(sample)

        # Sort by problem_id, sample_id
        samples.sort(key=lambda s: (s['problem_id'], s['sample_id']))
        return samples


class EvalStructureDetector:
    """Automatically detect evaluation directory structure."""

    def __init__(self):
        # Order matters: try multi-turn first, then single-turn
        self.loaders = [
            MultiTurnKernelLoader(),
            SingleTurnLoader(),
        ]

    def detect_and_load(self, eval_dir):
        """Auto-detect structure and return (loader, samples)."""
        for loader in self.loaders:
            if loader.is_applicable(eval_dir):
                samples = loader.load_samples(eval_dir)
                return loader, samples
        raise ValueError(f"Unknown evaluation directory structure in {eval_dir}")


# --- Metrics Formatter Classes ---

class MetricsFormatter(ABC):
    """Format evaluation metrics for display."""

    def format_for_json_display(self, eval_dict):
        """Format for JSON component (exclude fields, etc.)"""
        return {k: v for k, v in eval_dict.items()
                if not k.startswith('_') and k != 'prompt'}

    @abstractmethod
    def format_for_table(self, eval_dict):
        """Format single eval result for table row."""
        raise NotImplementedError

    @abstractmethod
    def build_comparison_table(self, turns_data):
        """Build pandas DataFrame for turn comparison."""
        raise NotImplementedError


class KernelMetricsFormatter(MetricsFormatter):
    """Formatter for kernel evaluation metrics."""

    TABLE_COLUMNS = [
        'turn_id', 'score', 'correctness', 'performance', 'compilation',
        'is_speedup_positive', 'is_decoy_kernel', 'num_custom_kernel',
        'time_coverage', 'finish_reason', 'error'
    ]

    def format_for_table(self, eval_dict):
        """Format single eval result for table row."""
        row = {}
        for col in self.TABLE_COLUMNS:
            value = eval_dict.get(col, '')
            if col in ['correctness', 'compilation', 'is_speedup_positive', 'is_decoy_kernel']:
                row[col] = format_boolean(value)
            elif col in ['performance', 'score']:
                if isinstance(value, (int, float)):
                    row[col] = f"{value:.2f}"
                else:
                    row[col] = str(value)
            elif col == 'time_coverage':
                row[col] = format_percentage(value)
            elif col == 'error':
                if value:
                    row[col] = str(value)[:50] + ('...' if len(str(value)) > 50 else '')
                else:
                    row[col] = ''
            else:
                row[col] = value
        return row

    def build_comparison_table(self, turns_data):
        """Build pandas DataFrame for turn comparison."""
        rows = []
        for turn in turns_data:
            row = self.format_for_table(turn['eval'])
            rows.append(row)

        df = pd.DataFrame(rows)
        # Reorder columns if they exist
        existing_cols = [col for col in self.TABLE_COLUMNS if col in df.columns]
        return df[existing_cols]


# --- UI Builder Classes ---

class GradioUIBuilder(ABC):
    """Abstract base class for building Gradio UI."""

    def __init__(self, config=None):
        self.config = config or VisualizerConfig()

    @abstractmethod
    def build_ui(self, samples):
        """Returns gr.Blocks() object."""
        raise NotImplementedError

    def format_sample_choice(self, sample):
        """Format sample for dropdown display."""
        if sample.get('is_multi_turn'):
            score = sample.get('total_score', 0.0)
            num_turns = sample.get('num_turns', 0)
            return f"Problem {sample['problem_id']} Sample {sample['sample_id']} (Score: {score:.2f}, {num_turns} turns)"
        else:
            score = sample.get('score', 0.0)
            return f"Problem {sample['problem_id']} Sample {sample['sample_id']} (Score: {score:.2f})"


class SingleTurnUIBuilder(GradioUIBuilder):
    """Builds single-turn visualization UI."""

    def build_ui(self, samples):
        """Build Gradio UI for single-turn visualization."""
        import gradio as gr

        # Create choices for dropdown
        choices = [self.format_sample_choice(s) for s in samples]

        def get_sample_details(choice):
            if not choice:
                return "", "", "", {}
            try:
                idx = choices.index(choice)
                s = samples[idx]

                code = s.get('_code', '')
                ref_code = s.get('_ref', '')
                prompt = s.get('prompt', 'No prompt saved.')

                # Prepare metrics JSON (exclude internal keys and large fields)
                display_json = {k: v for k, v in s.items()
                                if not k.startswith('_') and k != 'prompt'}

                return code, ref_code, prompt, display_json
            except ValueError:
                return "", "", "", {}

        with gr.Blocks(title="Kernel Evaluation Visualizer") as demo:
            gr.Markdown("## Kernel Evaluation Results Preview")

            with gr.Row():
                selector = gr.Dropdown(choices=choices, label="Select Sample",
                                       value=choices[0] if choices else None, interactive=True)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Evaluation Metrics")
                    json_out = gr.JSON(label="Metrics")
                with gr.Column(scale=4):
                    gr.Markdown("### Code Comparison")
                    with gr.Row():
                        code_out = gr.Code(language="python", label="Generated Kernel", interactive=False)
                        ref_code_out = gr.Code(language="python", label="Reference Code", interactive=False)

            with gr.Row():
                with gr.Accordion("Prompt / Input", open=False):
                    prompt_out = gr.Markdown()

            selector.change(fn=get_sample_details, inputs=selector,
                            outputs=[code_out, ref_code_out, prompt_out, json_out])

            # Initial load
            if choices:
                c, r, p, j = get_sample_details(choices[0])
                json_out.value = j
                code_out.value = c
                ref_code_out.value = r
                prompt_out.value = p

        return demo


class MultiTurnUIBuilder(GradioUIBuilder):
    """Builds multi-turn visualization UI with sidebar + tabs."""

    def __init__(self, config=None, metrics_formatter=None):
        super().__init__(config)
        self.metrics_formatter = metrics_formatter or KernelMetricsFormatter()

    def build_ui(self, samples):
        """Build Gradio UI for multi-turn visualization."""
        import gradio as gr

        # Create choices for dropdown
        choices = [self.format_sample_choice(s) for s in samples]

        def get_multi_turn_details(choice):
            if not choice:
                return "", [], None
            try:
                idx = choices.index(choice)
                sample = samples[idx]

                # Conversation text
                conversation_text = sample.get('conversation_text', 'Conversation history not available')

                # Build turn tabs content - return list of (code, json) tuples
                turns_content = []
                for turn in sample['turns']:
                    code = turn['code']
                    eval_json = self.metrics_formatter.format_for_json_display(turn['eval'])
                    turns_content.append((code, eval_json))

                # Build comparison table
                df = self.metrics_formatter.build_comparison_table(sample['turns'])

                return conversation_text, turns_content, df, sample['reference_code']
            except Exception as e:
                print(f"Error in get_multi_turn_details: {e}")
                return "", [], None, ""

        with gr.Blocks(title="Kernel Multi-Turn Evaluation Visualizer") as demo:
            gr.Markdown("## Kernel Evaluation Results Preview (Multi-Turn)")

            with gr.Row():
                selector = gr.Dropdown(choices=choices, label="Select Sample",
                                       value=choices[0] if choices else None, interactive=True)

            with gr.Row():
                # Left sidebar - Conversation history
                with gr.Column(scale=self.config.sidebar_scale):
                    gr.Markdown("### Full Conversation")
                    # 使用 Markdown 组件支持格式化显示
                    conversation_box = gr.Markdown(
                        value="",
                        label="Conversation",
                        height=600  # 固定高度，支持滚动
                    )

                # Right main area - Turn tabs
                with gr.Column(scale=self.config.main_area_scale):
                    with gr.Tabs() as tabs:
                        # Reference tab
                        with gr.Tab("Reference"):
                            ref_code_display = gr.Code(language="python", label="Reference Code", interactive=False)

                        # Pre-allocate turn tabs (up to max_tabs)
                        turn_tabs = []
                        for i in range(1, self.config.max_tabs + 1):
                            with gr.Tab(f"Turn {i}", visible=False) as turn_tab:
                                turn_code = gr.Code(language="python", label=f"Turn {i} Code", interactive=False)
                                turn_json = gr.JSON(label=f"Turn {i} Metrics")
                                turn_tabs.append((turn_tab, turn_code, turn_json))

            # Comparison table
            with gr.Row():
                gr.Markdown("### Turn Comparison Table")
            with gr.Row():
                comparison_table = gr.Dataframe(label="Metrics Across Turns", interactive=False)

            def update_ui(choice):
                """Update all UI components when sample changes."""
                if not choice:
                    return [""] + [gr.update(visible=False)] * (self.config.max_tabs * 3) + [None, ""]

                conv_text, turns_content, df, ref_code = get_multi_turn_details(choice)

                # Prepare outputs
                outputs = [conv_text]  # conversation_box

                # Update turn tabs
                for i in range(self.config.max_tabs):
                    if i < len(turns_content):
                        # Show this turn tab
                        code, eval_json = turns_content[i]
                        outputs.extend([
                            gr.update(visible=True),  # turn_tab visibility
                            code,  # turn_code
                            eval_json  # turn_json
                        ])
                    else:
                        # Hide this turn tab
                        outputs.extend([
                            gr.update(visible=False),  # turn_tab visibility
                            "",  # turn_code
                            {}  # turn_json
                        ])

                outputs.append(df)  # comparison_table
                outputs.append(ref_code)  # ref_code_display

                return outputs

            # Wire up the change event
            turn_tab_components = []
            for tab, code, json_comp in turn_tabs:
                turn_tab_components.extend([tab, code, json_comp])

            selector.change(
                fn=update_ui,
                inputs=selector,
                outputs=[conversation_box] + turn_tab_components + [comparison_table, ref_code_display]
            )

            # Initial load
            if choices:
                initial_outputs = update_ui(choices[0])
                conversation_box.value = initial_outputs[0]
                comparison_table.value = initial_outputs[-2]
                ref_code_display.value = initial_outputs[-1]
                # Update turn tabs
                for i, (tab, code, json_comp) in enumerate(turn_tabs):
                    idx = 1 + i * 3
                    if idx < len(initial_outputs) - 2:
                        code.value = initial_outputs[idx + 1]
                        json_comp.value = initial_outputs[idx + 2]

        return demo


def launch_gradio_visualizer(eval_dir, share=False):
    """
    Launch a Gradio interface to visualize evaluation results.

    This function automatically detects whether the evaluation directory contains
    single-turn or multi-turn results and launches the appropriate UI.

    Modular design allows easy extension to other task types (math, code, etc.).
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please install it with `pip install gradio`.")
        return

    print(f"Loading evaluation results from {eval_dir}...")

    try:
        # 1. Auto-detect structure and load data
        detector = EvalStructureDetector()
        loader, samples = detector.detect_and_load(eval_dir)

        if not samples:
            print(f"No evaluation results found in {eval_dir}")
            return

        print(f"Found {len(samples)} samples.")

        # 2. Select appropriate UI builder based on loader type
        if isinstance(loader, MultiTurnKernelLoader):
            print("Detected multi-turn structure. Launching multi-turn visualizer...")
            config = VisualizerConfig()
            metrics_formatter = KernelMetricsFormatter()
            ui_builder = MultiTurnUIBuilder(config=config, metrics_formatter=metrics_formatter)
        else:
            print("Detected single-turn structure. Launching single-turn visualizer...")
            ui_builder = SingleTurnUIBuilder()

        # 3. Build and launch UI
        demo = ui_builder.build_ui(samples)

        print(f"Launching Gradio server...")
        # inbrowser=False 避免在 SSH 环境中打开 lynx 等文本浏览器
        app, local_url, share_url = demo.launch(server_name="0.0.0.0", share=share, inbrowser=False)

        # 4. Save Gradio URLs to local file
        url_file_path = os.path.join(eval_dir, "gradio_url.txt")
        try:
            with open(url_file_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("Gradio Visualization URLs\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Local URL:  {local_url}\n")
                if share and share_url:
                    f.write(f"Public URL: {share_url}\n")
                    f.write("\nNote: Public URL expires after 72 hours\n")
                f.write("\n" + "=" * 80 + "\n")
            print(f"\n✓ URLs saved to: {url_file_path}")
        except Exception as e:
            print(f"Warning: Could not save URLs to file: {e}")

        # 输出醒目的 URL 信息
        print("\n" + "=" * 80)
        print("🌐 Gradio Visualization Server Started")
        print("=" * 80)
        print(f"\nLocal URL:  {local_url}")
        if share and share_url:
            print(f"Public URL: {share_url}")
            print("\n⚠️  Public URL expires after 72 hours")
        print(f"\n📁 URLs also saved to: {url_file_path}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"Error launching visualizer: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(config_path='config', config_name='kernel_grading', version_base=None)
def main(config):
    # Check for visualization only mode
    if config.get("visualize_only", False):
        print("=" * 80)
        print("VISUALIZATION ONLY MODE")
        print("=" * 80)
        
        target_dir = config.get("visualize_dir")
        print(f"target_dir: {target_dir}")
        print(f"config.get('visualize_dir'): {config.get('visualize_dir')}")
        if not target_dir:
            print(f"target_dir is None")
            # Try to infer from data.output_path
            if "data" in config and "output_path" in config.data:
                output_path = config.data.output_path
                target_dir = os.path.join(os.path.dirname(output_path), "eval_outputs")
                print(f"target_dir: {target_dir}")
                print(f"output_path: {output_path}")
                print(f"os.path.dirname(output_path): {os.path.dirname(output_path)}")
                print(f"os.path.join(os.path.dirname(output_path), 'eval_outputs'): {os.path.join(os.path.dirname(output_path), 'eval_outputs')}")

        if target_dir and os.path.exists(target_dir):
            print(f"Visualizing results from: {target_dir}")
            share = config.get("gradio_share", False)
            launch_gradio_visualizer(target_dir, share=share)
            return
        else:
            print(f"Error: Could not find evaluation artifacts directory.")
            print(f"Checked path: {target_dir}")
            print("Please provide 'visualize_dir=/path/to/eval_outputs' or ensure 'data.output_path' points to the run location.")
            return

    # Validate critical config sections exist
    from omegaconf import OmegaConf
    print("=" * 80)
    print("CONFIG VALIDATION")
    print("=" * 80)

    if not OmegaConf.is_dict(config):
        raise ValueError("Config is not a dict!")

    # Check critical sections
    required_sections = ['data', 'model', 'actor_rollout_ref', 'reward_model', 'trainer']
    missing = [s for s in required_sections if s not in config]
    if missing:
        print(f"ERROR: Missing required config sections: {missing}")
        print(f"Available sections: {list(config.keys())}")
        raise ValueError(f"Missing required config sections: {missing}")

    # Check actor config specifically
    if 'actor' not in config.actor_rollout_ref:
        print("ERROR: actor_rollout_ref.actor is missing!")
        print(f"actor_rollout_ref keys: {list(config.actor_rollout_ref.keys())}")
        raise ValueError("actor_rollout_ref.actor configuration is missing")

    print("✓ Config validation passed")
    print("=" * 80)

    # Run generation and get output directory
    eval_output_dir = run_generation(config)
    
    # Check if gradio is enabled
    if config.get("gradio", False) and eval_output_dir:
        share = config.get("gradio_share", False)
        launch_gradio_visualizer(eval_output_dir, share=share)


def run_generation(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    return ray.get(main_task.remote(config))


# ============================================================================
# Helper Functions (moved from main_task for better code organization)
# ============================================================================

def extract_code(text):
    """Extract Python code from text (handles markdown code blocks)."""
    if not text:
        return ""
    # try markdown python block
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _coerce_extra_metric_value(value: Any) -> Optional[float]:
    """Convert heterogeneous reward extra info entries into scalars when possible."""
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        return float(value)

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return None

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _coerce_extra_metric_value(value[0])
        return None

    return None


@ray.remote(num_cpus=1)
def main_task(config):
    """
    A Ray remote task that handles both generation of kernel code responses and scoring to produce solve rates.
    """
    save_path = os.path.dirname(config.data.output_path)
    if os.path.isdir(save_path):
        print(f"The save path '{save_path}' exists.")
    else:
        print(f"The save path '{save_path}' does not exist. Creating...")
        makedirs(save_path, exist_ok=True)
        print(f"Directory '{save_path}' created.")

    raw_response_path = config.data.get('raw_response_path')
    dataproto_path = config.data.get('dataproto_path')

    from pprint import pprint

    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # DEBUG: Check if actor config exists
    print("=" * 80)
    print("DEBUG: Checking config structure...")
    print(f"Has actor_rollout_ref: {hasattr(config, 'actor_rollout_ref')}")
    if hasattr(config, 'actor_rollout_ref'):
        print(f"Has actor_rollout_ref.actor: {hasattr(config.actor_rollout_ref, 'actor')}")
        if hasattr(config.actor_rollout_ref, 'actor'):
            print(f"actor.fsdp_config.fsdp_size: {config.actor_rollout_ref.actor.fsdp_config.fsdp_size}")
    print("=" * 80)

    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer

    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # Apply Qwen3 chat template fix if needed
    if config.trainer.get("fix_qwen3_chat_template", False):
        print(f"Fixing Qwen3 chat template for {local_path}...")
        tokenizer.chat_template = QWEN3CHATTEMPLATE

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    rollout_mode = config.actor_rollout_ref.rollout.get('mode', 'sync')
    async_rollout_mode = rollout_mode in ("async_vllm", "async_agent", "standalone_vllm")

    # Check if multi-turn is enabled
    multi_turn_enabled = (
        hasattr(config.actor_rollout_ref, 'rollout') and
        hasattr(config.actor_rollout_ref.rollout, 'multi_turn') and
        config.actor_rollout_ref.rollout.multi_turn.get('enable', False)
    )

    if multi_turn_enabled:
        print("=" * 80)
        print("MULTI-TURN MODE ENABLED")
        max_user_turns = config.actor_rollout_ref.rollout.multi_turn.get('max_user_turns', 'N/A')
        print(f"Max user turns per iteration: {max_user_turns}")

        # Check if multi-iteration is enabled
        multi_iter_config = config.actor_rollout_ref.rollout.multi_turn.get('multi_iteration', {})
        if multi_iter_config.get('enable', False):
            max_iterations = multi_iter_config.get('max_iterations', 1)
            remain_turns = multi_iter_config.get('remain_turns', 2)
            iteration_method = multi_iter_config.get('iteration_method', 'last')

            print("MULTI-ITERATION ENABLED")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Remain turns: {remain_turns}")
            print(f"  Iteration method: {iteration_method}")

            if max_iterations > 1 and isinstance(max_user_turns, int):
                total_turns = max_user_turns + (max_iterations - 1) * (max_user_turns - remain_turns)
                print(f"  Expected total turns: {total_turns}")

        print("=" * 80)

    # Initialize reward manager (shared with async rollout and scoring)
    print("Initializing reward manager for grading...")
    compute_score = get_custom_reward_fn(config)
    try:
        server_url = config.reward_model.server_url
        if server_url and server_url not in ['sandbox', 'null', 'None']:
            _check_kernel_server_health(server_url)
    except Exception as e:
        print(f"Warning: Kernel server health check failed: {e}")
    
    # Check for Gradio flag in config
    enable_gradio = config.get("gradio", False)
    if enable_gradio:
        print("Gradio visualization enabled.")

    # Initialize wandb if specified in config
    use_wandb = 'wandb' in config.trainer.get('logger', [])
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.trainer.get('project_name', 'kernel-grading'),
                name=config.trainer.get('experiment_name', 'grading'),
                config=dict(config),
                resume='allow'
            )
            print("✅ Wandb initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            use_wandb = False

    # Initialize validation generations logger for wandb/swanlab
    validation_logger = ValidationGenerationsLogger()

    reward_manager_cls = get_reward_manager_cls(config)
    reward_fn_kwargs = {
        'tokenizer': tokenizer,
        'num_examine': 5,
        'compute_score': compute_score,
    }
    try:
        reward_fn = reward_manager_cls(
            **reward_fn_kwargs,
            reward_fn_key=config.data.get('reward_fn_key', None),
            reward_config=config.reward_model,
            is_valid=True,
        )
    except TypeError:
        reward_fn = reward_manager_cls(**{**reward_fn_kwargs, 'num_examine': 0})

    # use mock processor
    class MockProcessor:
        image_processor = type('', (), {'return_tensors': 'pt'})()

    dataset = RLHFDataset(
        parquet_files=config.data.path,
        tokenizer=tokenizer,
        processor=MockProcessor(),
        prompt_key=config.data.prompt_key,
        image_key=config.data.get('image_key', 'images'),
        max_prompt_length=config.data.max_prompt_length,
        filter_prompts=True,
        apply_chat_template=config.data.apply_chat_template,
        return_raw_chat=async_rollout_mode,
        truncation='error',
        filter_overlong_prompts=config.data.filter_overlong_prompts,
    )

    dataloader = StatefulDataLoader(
        dataset=dataset,
        # Validation datasets are sent to inference engines as a whole batch,
        # which will schedule the memory themselves.
        batch_size=config.data.batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # if dataproto_path exist, directly load it
    if dataproto_path and os.path.exists(dataproto_path):
        # read out the existing raw responses
        print(f"Load DataProto from {dataproto_path}...")
        all_dataproto = DataProto.load_from_disk(dataproto_path)
        all_input_texts, all_output_texts = [], []  # Will be populated if needed
    else:  # otherwise, generate responses
        actor_rollout_config = config.actor_rollout_ref
        wg = None
        async_rollout_manager = None

        if rollout_mode == "standalone_vllm":
            from kernel.workers.rollout.async_server import StandaloneVLLMEngineManager

            total_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
            async_rollout_manager = StandaloneVLLMEngineManager(
                config=actor_rollout_config,
                tokenizer=tokenizer,
                reward_fn=reward_fn,
                val_reward_fn=reward_fn,
                total_gpus=total_gpus,
            )
            print("StandaloneVLLMEngineManager initialized")
        else:
            # Determine worker class based on rollout mode
            from verl_patch.workers.code.fsdp_workers import AsyncActorRolloutRefWorker

            if rollout_mode in ("async_vllm", "async_agent"):
                actor_rollout_cls = AsyncActorRolloutRefWorker
                print(f"Using async rollout mode: {rollout_mode}")
            else:
                actor_rollout_cls = ActorRolloutRefWorker
                print(f"Using sync rollout mode")

            ray_cls_with_init = RayClassWithInitArgs(
                cls=ray.remote(actor_rollout_cls), config=actor_rollout_config, role='rollout'
            )
            resource_pool = RayResourcePool(
                process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix="global_pool",
            )
            wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
            wg.init_model()

            print("Ray namespace:", ray.get_runtime_context().namespace)
            print("Known actors:", ray.util.list_named_actors())
            print("AsyncActorRolloutRefWorker world_size:", wg.world_size)

            if rollout_mode == "async_vllm":
                from kernel.workers.rollout.async_server import AsyncLLMEngineManager
                async_rollout_manager = AsyncLLMEngineManager(
                    config=actor_rollout_config,
                    worker_group=wg,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    val_reward_fn=reward_fn,
                )
                print("AsyncLLMEngineManager initialized")
            elif rollout_mode == "async_agent":
                from verl_patch.experimental.agent_loop.agent_loop import AgentLoopManager
                async_rollout_manager = AgentLoopManager(
                    config=config,
                    worker_group=wg,
                )
                print("AgentLoopManager initialized")

        # Setup for detailed output saving
    output_dir = os.path.dirname(config.data.output_path)
    eval_output_dir = os.path.join(output_dir, "eval_outputs")
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"Detailed evaluation artifacts will be saved to: {eval_output_dir}")

    # Accumulators for metrics
    reward_tensor_lst = []
    data_source_lst = []
    reward_extra_info_dict = defaultdict(list)
    problem_sample_counts = defaultdict(int)

    # Track problem_id to uid mapping for multi-turn mode
    # Maps problem_id -> list of (uid, sample_id) tuples to preserve order
    problem_to_conversations = defaultdict(list)

    # Cache for complete messages per uid (for multi-turn)
    uid_messages_cache = {}

    # Fix 2: Global sample counter to ensure deterministic sample_id across batches
    global_sample_counter = defaultdict(int)  # Maps problem_id -> next_available_sample_id

    def process_batch(batch, batch_input_texts=None, batch_offset=0):
        def json_safe(val):
            """Best-effort conversion to JSON serializable python types."""
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    return val.item()
                return val.detach().cpu().tolist()
            if isinstance(val, (np.integer, np.floating, np.bool_)):
                return val.item()
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        def get_seq_item(seq, idx):
            """Helper to safely pull an item from list/array-like structures."""
            if seq is None:
                return None
            try:
                if isinstance(seq, np.ndarray):
                    seq = seq.tolist()
            except Exception:
                pass
            try:
                return seq[idx]
            except Exception:
                return None

        def extract_reward_info_for_sample(extra_info, idx):
            """Build per-sample reward_extra_info dictionary."""
            per_sample = {}
            if not extra_info:
                return per_sample
            for key, raw_val in extra_info.items():
                picked = get_seq_item(raw_val, idx)
                if picked is None:
                    picked = raw_val
                per_sample[key] = json_safe(picked)
            return per_sample

        # Detect if this is multi-turn mode
        has_turn_indices = 'turn_indices' in batch.batch
        is_multi_turn = has_turn_indices and multi_turn_enabled

        print(f"[DEBUG] Process Batch - saving artifacts...")

        # Extract pre-computed reward tensor (rewards are already computed in generation loop)
        if 'token_level_scores' in batch.batch:
            reward_tensor = batch.batch['token_level_scores']
        else:
            # If somehow not present, create zero tensor as placeholder
            batch_size = len(batch)
            reward_tensor = torch.zeros(batch_size, 1)

        # Save artifacts (code and eval json)
        batch_size = len(batch)

        # Get reward_extra_info from batch for per-sample eval info
        batch_reward_extra_info = batch.non_tensor_batch.get('reward_extra_info')
        if batch_reward_extra_info is None:
            batch_reward_extra_info_list = [{}] * batch_size
        elif hasattr(batch_reward_extra_info, 'tolist'):
            batch_reward_extra_info_list = batch_reward_extra_info.tolist()
        else:
            batch_reward_extra_info_list = list(batch_reward_extra_info)

        pid_uid_to_sample_id = {}
        last_global_turn_idx = {}

        for i in range(batch_size):
            # Resolve Problem ID
            pid = None
            if "extra_info" in batch.non_tensor_batch:
                extra_infos = batch.non_tensor_batch["extra_info"]
                if isinstance(extra_infos, list) and len(extra_infos) > i:
                    item_extra = extra_infos[i]
                    if isinstance(item_extra, dict):
                            pid = item_extra.get("problem_id")

            if pid is None:
                    p_indices = batch.non_tensor_batch.get('prompt_index')
                    pid = p_indices[i] if p_indices is not None else f"batch_unk_idx{i}"

            if pid not in pid_uid_to_sample_id:
                pid_uid_to_sample_id[pid] = {}

            uid = batch.non_tensor_batch["uid"][i]

            if uid not in pid_uid_to_sample_id[pid]:
                # Fix 2: Use global counter instead of per-batch counter
                # This ensures sample_id is deterministic across all batches
                sample_id = global_sample_counter[pid]
                global_sample_counter[pid] += 1
                pid_uid_to_sample_id[pid][uid] = sample_id

            # Resolve Sample ID
            sid = pid_uid_to_sample_id[pid][uid]
            
            # Get Code
            response_ids = batch.batch['responses'][i]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            code = extract_code(response_text)
            
            # Get Prompt (Try to recover if possible)
            prompt_text = ""
            if batch_input_texts and len(batch_input_texts) > i:
                    prompt_text = batch_input_texts[i]
            elif "input_ids" in batch.batch:
                try:
                    p_ids = batch.batch["input_ids"][i]
                    prompt_text = tokenizer.decode(p_ids, skip_special_tokens=True)
                except:
                    pass
            
            # Get Reference Code (robust fallback across potential schema variations)
            ref_code = None
            rm_batch = batch.non_tensor_batch.get("reward_model")
            item_rm = get_seq_item(rm_batch, i)
            if isinstance(item_rm, str):
                try:
                    parsed_rm = json.loads(item_rm)
                    if isinstance(parsed_rm, dict):
                        item_rm = parsed_rm
                except Exception:
                    pass
            if isinstance(item_rm, dict):
                ref_code = (
                    item_rm.get("ground_truth")
                    or item_rm.get("reference_code")
                    or item_rm.get("reference")
                )
            elif item_rm:
                ref_code = item_rm

            if not ref_code:
                ref_batch = batch.non_tensor_batch.get("reference_code")
                ref_code = get_seq_item(ref_batch, i)
            if not ref_code:
                ref_batch = batch.non_tensor_batch.get("reference")
                ref_code = get_seq_item(ref_batch, i)
            if not ref_code:
                gt_batch = batch.non_tensor_batch.get("ground_truth")
                ref_code = get_seq_item(gt_batch, i)
            if not ref_code and "extra_info" in batch.non_tensor_batch:
                extra_item = get_seq_item(batch.non_tensor_batch["extra_info"], i)
                if isinstance(extra_item, dict):
                    ref_code = extra_item.get("reference_code") or extra_item.get("reference")
            # For multi-turn mode, extract reference from first user message
            if not ref_code and is_multi_turn:
                multiturn_messages = batch.non_tensor_batch.get('multiturn_messages', None)
                if multiturn_messages is not None:
                    # Find the first turn row for this sample
                    for j in range(len(batch.batch)):
                        if batch.non_tensor_batch['uid'][j] == batch.non_tensor_batch['uid'][i]:
                            j_turn = int(batch.batch['turn_indices'][j].item())
                            if j_turn == 0:
                                complete_messages = get_seq_item(multiturn_messages, j)
                                if complete_messages and isinstance(complete_messages, list) and len(complete_messages) > 0:
                                    # Extract first user message (messages[0])
                                    first_user_msg = complete_messages[0]
                                    if isinstance(first_user_msg, dict) and first_user_msg.get('role') == 'user':
                                        first_user_content = first_user_msg.get('content', '')
                                        # Find all python code blocks
                                        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", first_user_content, re.DOTALL)
                                        if code_blocks:
                                            # Use the last code block as reference
                                            ref_code = code_blocks[-1].strip()
                                break
            
            # sample_inputs = []
            # sample_outputs = []
            # if is_multi_turn:
            #     multiturn_messages = batch.non_tensor_batch.get('multiturn_messages')

            #     sample_first_turn = {}  # sample_id -> row_idx of first turn
            #     for j in range(len(batch.batch)):
            #         s_idx = batch.non_tensor_batch['uid'][j]
            #         t_idx = int(batch.batch['turn_indices'][j])

            #         if t_idx == -1:  # Skip padding turns
            #             continue

            #         # First turn has the messages
            #         if s_idx not in sample_first_turn:
            #             sample_first_turn[s_idx] = j

                # Build input/output for each sample using multiturn_messages
                # for s_idx in sorted(sample_first_turn.keys()):
                #     first_idx = sample_first_turn[s_idx]

                #     if multiturn_messages is not None and multiturn_messages[first_idx] is not None:
                #         messages = multiturn_messages[first_idx]

                #         # Input: extract first user message
                #         first_user_msg = ""
                #         for msg in messages:
                #             if msg.get('role') == 'user':
                #                 first_user_msg = msg.get('content', '')
                #                 break
                #         sample_inputs.append(first_user_msg)

                #         # Output: build complete conversation string
                #         full_output = ""
                #         for msg in messages:
                #             role = msg.get('role', 'unknown')
                #             content = msg.get('content', '')
                #             full_output += f"[{role}]\n{content}\n\n"
                #         sample_outputs.append(full_output)
                #     else:
                #         sample_inputs.append("[No messages available]")
                #         sample_outputs.append("[No messages available]")

            # Fallback: extract from prompt_text
            if not ref_code and prompt_text:
                code_blocks = re.findall(r"```(?:python)?\n(.*?)```", prompt_text, re.DOTALL)
                if code_blocks:
                    ref_code = code_blocks[-1].strip()

            if isinstance(ref_code, (dict, list)):
                try:
                    ref_code_str = json.dumps(ref_code, ensure_ascii=False, indent=2)
                except Exception:
                    ref_code_str = str(ref_code)
            else:
                ref_code_str = ref_code or ""
            if not str(ref_code_str).strip():
                ref_code_str = "# Reference code not available"

            # Get Eval Info for this sample - use per-sample reward_extra_info
            sample_eval = {}

            # Extract reward_extra_info for this sample
            if i < len(batch_reward_extra_info_list):
                sample_reward_info = batch_reward_extra_info_list[i]
                if sample_reward_info and isinstance(sample_reward_info, dict):
                    # Keep flattened fields for backward compatibility
                    sample_eval.update(sample_reward_info)
                    # Also preserve a nested copy for easier inspection
                    sample_eval["reward_extra_info"] = sample_reward_info

            sample_eval["score"] = reward_tensor[i].sum().item()
            sample_eval["problem_id"] = pid
            sample_eval["sample_id"] = sid
            sample_eval["prompt"] = prompt_text

            # Add multi-turn specific fields if applicable
            if is_multi_turn:
                turn_indices = batch.batch['turn_indices'].cpu().numpy()
                uids = batch.non_tensor_batch['uid']
                global_turn_indices = batch.non_tensor_batch.get('global_turn_indices', None)
                turn_id = int(turn_indices[i])
                uid = uids[i]

                # Skip padding turns
                if turn_id == -1:
                    continue

                sample_eval["turn_id"] = turn_id
                sample_eval["uid"] = uid

                # Attach global turn index if provided (multi-iteration ordering)
                global_idx = get_seq_item(global_turn_indices, i)
                if global_idx is not None:
                    try:
                        global_idx_val = int(global_idx)
                        sample_eval["global_turn_idx"] = global_idx_val
                        if uid in last_global_turn_idx and global_idx_val < last_global_turn_idx[uid]:
                            print(
                                f"Warning: non-monotonic global_turn_idx for uid {uid}: "
                                f"{global_idx_val} < {last_global_turn_idx[uid]}"
                            )
                        last_global_turn_idx[uid] = global_idx_val
                    except Exception:
                        pass

                # Add finish reason if available
                if 'finish_reasons' in batch.non_tensor_batch:
                    finish_reasons = batch.non_tensor_batch['finish_reasons']
                    sample_eval["finish_reason"] = get_seq_item(finish_reasons, i)

                # Cache complete messages for this uid (only available in turn_id==0 row)
                multiturn_messages = batch.non_tensor_batch.get('multiturn_messages', None)
                if multiturn_messages is not None:
                    current_messages = get_seq_item(multiturn_messages, i)
                    # turn_id is 1-based in async engines; tolerate 0-based for older paths.
                    if current_messages is not None and turn_id in (0, 1):
                        # Cache the complete messages for this uid
                        uid_messages_cache[uid] = current_messages

            # Write files - use different structure for multi-turn vs single-turn
            if is_multi_turn:
                # Track problem_id to uid mapping (only record the first seen turn for each uid)
                if (pid, uid) not in [(p, u) for p, u in problem_to_conversations[pid]]:
                    problem_to_conversations[pid].append((uid, sid))

                # Multi-turn: Save in problem_X_sample_Y directory
                conv_dir = os.path.join(eval_output_dir, f"problem_{pid}_sample_{sid}")
                os.makedirs(conv_dir, exist_ok=True)

                # Save turn-specific files
                # print(f"[DEBUG] Saving turn {turn_id} for sample {sid} with uid {uid}")
                with open(os.path.join(conv_dir, f"turn_{turn_id}_kernel.py"), "w") as f:
                    f.write(code)
                with open(os.path.join(conv_dir, f"turn_{turn_id}_eval.json"), "w") as f:
                    json.dump(sample_eval, f, indent=2, default=_json_default)

                # Save messages state for this turn (if available)
                # Extract the relevant slice of messages for this specific turn
                # turn_i corresponds to messages[2*(i-1) : 2*(i-1)+3] = [user_i, response_i, user_{i+1}]
                # where user_{i+1} is the evaluation feedback for response_i
                complete_messages = uid_messages_cache.get(uid, None)
                if complete_messages is not None and isinstance(complete_messages, list):
                    # Extract slice for this turn: [user_i, response_i, user_{i+1}]
                    turn_offset = turn_id - 1 if turn_id > 0 else 0
                    start_idx = 2 * turn_offset
                    end_idx = 2 * turn_offset + 3
                    turn_messages_slice = complete_messages[start_idx:end_idx]

                    if len(turn_messages_slice) > 0:
                        messages_state_file = os.path.join(conv_dir, f"turn_{turn_id}_state.json")
                        with open(messages_state_file, "w") as f:
                            json.dump({
                                "turn_id": turn_id,
                                "messages": turn_messages_slice,
                                "uid": uid,
                                "problem_id": pid,
                                "sample_id": sid
                            }, f, indent=2, default=_json_default)

                # Save reference code in conversation root (same for all turns)
                ref_file_path = os.path.join(conv_dir, "reference.py")
                if not os.path.exists(ref_file_path):
                    with open(ref_file_path, "w") as f:
                        f.write(str(ref_code_str))
            else:
                # Single-turn: Use original flat structure
                base_name = f"problem_{pid}_sample_{sid}"
                with open(os.path.join(eval_output_dir, f"{base_name}_kernel.py"), "w") as f:
                    f.write(code)
                with open(os.path.join(eval_output_dir, f"{base_name}_ref.py"), "w") as f:
                    f.write(str(ref_code_str))
                with open(os.path.join(eval_output_dir, f"{base_name}_eval.json"), "w") as f:
                    json.dump(sample_eval, f, indent=2, default=_json_default)
        # else:
            # DataProto style
            # reward_tensor = reward_result["reward_tensor"]
            # cur_data_source = batch.non_tensor_batch.get('data_source', ['kernel'] * reward_tensor.shape[0])

            # # Collect extra_info from non_tensor_batch
            # if "reward_extra_info" in batch.non_tensor_batch:
            #     reward_extra_infos = batch.non_tensor_batch["reward_extra_info"]
            #     for sample_extra, data_source in zip(reward_extra_infos, cur_data_source):
            #         if sample_extra is None:
            #             continue
            #         for key, extra_val in sample_extra.items():
            #             # unwrap single-element lists
            #             if isinstance(extra_val, list) and len(extra_val) == 1:
            #                 extra_val = extra_val[0]
            #             composed_key = f'{key}_{data_source}'
            #             reward_extra_info_dict[composed_key].append(extra_val)

    def generate_conversation_summaries(uid_to_conversation_dict):
        """Generate summary.json and full_conversation.txt files for each conversation directory.

        Fix 1: Uses UID-based lookup to ensure conversation content aligns with kernel code,
        eliminating dependency on filesystem ordering.

        Args:
            uid_to_conversation_dict: Dictionary mapping UID -> full conversation text
        """
        if not multi_turn_enabled:
            return

        print("Generating conversation summaries...")
        conv_dirs = [d for d in os.listdir(eval_output_dir) if d.startswith("problem_")]

        conversations_found = 0
        conversations_missing = 0

        for conv_dir_name in sorted(conv_dirs, key=_parse_problem_sample_dir):
            conv_dir = os.path.join(eval_output_dir, conv_dir_name)
            if not os.path.isdir(conv_dir):
                continue

            # Load all turn eval files to get metadata
            turn_files = sorted(
                [f for f in os.listdir(conv_dir) if f.startswith("turn_") and f.endswith("_eval.json")],
                key=_parse_turn_id_from_filename,
            )
            if not turn_files:
                continue

            turns_data = []
            total_score = 0.0
            for turn_file in turn_files:
                with open(os.path.join(conv_dir, turn_file)) as f:
                    turn_data = json.load(f)
                    turns_data.append(turn_data)
                    total_score += turn_data.get("score", 0.0)
            # Ensure turns are ordered by numeric turn_id from content (more robust than filenames)
            turns_data.sort(key=lambda t: int(t.get("turn_id", 0)))

            # Extract UID from turn_0 eval for lookup
            uid = turns_data[0].get("uid")
            if not uid:
                print(f"Warning: No uid found in {conv_dir_name}/first turn eval.json")
                uid = conv_dir_name.replace("conversation_", "")  # Fallback

            # Build summary
            summary = {
                "uid": uid,
                "num_turns": len(turns_data),
                "total_score": float(total_score),
                "per_turn_scores": [t.get("score", 0.0) for t in turns_data],
                "problem_id": turns_data[0].get("problem_id"),
            }

            # Add improvement tracking
            if len(turns_data) > 1:
                summary["improvement"] = {
                    "first_turn_score": turns_data[0].get("score", 0.0),
                    "last_turn_score": turns_data[-1].get("score", 0.0),
                    "improved": turns_data[-1].get("score", 0.0) > turns_data[0].get("score", 0.0),
                }

            # Save summary
            with open(os.path.join(conv_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2, default=_json_default)

            # Fix 1: Use UID-based lookup instead of linear indexing
            full_output = uid_to_conversation_dict.get(uid, "[No conversation data available]\n")

            if uid in uid_to_conversation_dict:
                conversations_found += 1
            else:
                conversations_missing += 1
                print(f"Warning: No conversation found for UID {uid} in directory {conv_dir_name}")

            # Save full conversation
            with open(os.path.join(conv_dir, "full_conversation.txt"), "w") as f:
                f.write(full_output)

        print(f"Generated summaries for {len(conv_dirs)} problems")
        print(f"  - Conversations found: {conversations_found}")
        print(f"  - Conversations missing: {conversations_missing}")

    def save_conversations_to_jsonl(all_dataproto, reward_tensor, tokenizer, output_path):
        """Save multi-turn conversations to simplified JSONL format."""
        if not multi_turn_enabled:
            return

        if 'turn_indices' not in all_dataproto.batch:
            print("Warning: turn_indices not found in batch, skipping JSONL conversation logging")
            return

        print("Saving conversations to JSONL...")
        turn_indices = all_dataproto.batch['turn_indices'].cpu().numpy()
        uids = all_dataproto.non_tensor_batch['uid']
        scores = reward_tensor.cpu().numpy()
        global_turn_indices = all_dataproto.non_tensor_batch.get('global_turn_indices', None)

        def _get_seq_item(seq, idx):
            if seq is None:
                return None
            try:
                if isinstance(seq, np.ndarray):
                    seq = seq.tolist()
            except Exception:
                pass
            try:
                return seq[idx]
            except Exception:
                return None

        # Get reward_extra_info if available
        reward_extra_info_raw = all_dataproto.non_tensor_batch.get('reward_extra_info', None)

        # Convert to list if it's a numpy array or other iterable
        if reward_extra_info_raw is None:
            reward_extra_info_list = []
        elif hasattr(reward_extra_info_raw, 'tolist'):
            reward_extra_info_list = reward_extra_info_raw.tolist()
        else:
            reward_extra_info_list = list(reward_extra_info_raw)

        # Group by conversation (uid)
        conversations = defaultdict(lambda: {'turns': [], 'total_score': 0.0})
        for i in range(len(all_dataproto.batch)):
            uid = uids[i]
            turn_id = int(turn_indices[i])
            if turn_id == -1:  # Skip padding
                continue

            global_idx = _get_seq_item(global_turn_indices, i)
            turn_entry = {
                'turn_id': turn_id,
                'response': tokenizer.decode(all_dataproto.batch['responses'][i], skip_special_tokens=True),
                'score': float(scores[i]),
            }
            if global_idx is not None:
                try:
                    turn_entry['global_turn_idx'] = int(global_idx)
                except Exception:
                    pass

            # Add reward extra info if available
            if len(reward_extra_info_list) > 0 and i < len(reward_extra_info_list):
                extra_info = reward_extra_info_list[i]
                if isinstance(extra_info, dict) and len(extra_info) > 0:
                    turn_entry['metrics'] = {k: _json_default(v) for k, v in extra_info.items()}

            conversations[uid]['turns'].append(turn_entry)
            conversations[uid]['total_score'] += scores[i]

        # Write to JSONL (one conversation per line)
        jsonl_path = output_path.replace('.parquet', '_conversations.jsonl').replace('.jsonl', '_conversations.jsonl')
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, 'w') as f:
            for uid, data in conversations.items():
                # Sort turns by global_turn_idx if available, else by turn_id
                if data['turns'] and 'global_turn_idx' in data['turns'][0]:
                    data['turns'].sort(key=lambda t: t.get('global_turn_idx', t['turn_id']))
                else:
                    data['turns'].sort(key=lambda t: t['turn_id'])
                entry = {
                    'uid': str(uid),
                    'num_turns': len(data['turns']),
                    'total_score': float(data['total_score']),
                    'turns': data['turns'],
                }
                f.write(json.dumps(entry, default=_json_default) + '\n')

        print(f"Saved {len(conversations)} conversations to {jsonl_path}")

    def _log_multiturn_to_jsonl(gen_batch_output: DataProto, scores: list, config, tokenizer):
        """Log multi-turn conversations to JSONL file (incremental append).

        Args:
            gen_batch_output: DataProto containing multi-turn conversation data
            scores: Reward scores for each row (flattened format)
            config: Configuration object with rollout settings
            tokenizer: Tokenizer for decoding prompts/responses
        """
        # Get the save path from config
        rollout_save_jsonl = config.actor_rollout_ref.rollout.multi_turn.rollout_save_jsonl

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(rollout_save_jsonl), exist_ok=True)

        # Get sample and turn indices
        turn_indices = gen_batch_output.batch["turn_indices"].cpu().numpy()
        uids = gen_batch_output.non_tensor_batch.get("uid", None)
        multiturn_messages = gen_batch_output.non_tensor_batch.get("multiturn_messages", None)

        # Find first turn (with messages) for each sample and aggregate scores
        sample_data = {}  # uid -> {'first_idx': row_idx, 'num_turns': int, 'total_score': float}
        for i in range(len(gen_batch_output.batch)):
            if uids is not None:
                s_idx = uids[i]
            else:
                raise ValueError("uids is None")

            t_idx = int(turn_indices[i])

            if t_idx == -1:  # Skip padding turns
                continue

            if s_idx not in sample_data:
                sample_data[s_idx] = {
                    "first_idx": i,
                    "num_turns": t_idx,
                    "total_score": scores[i],
                }
            else:
                sample_data[s_idx]["total_score"] += scores[i]
                if t_idx > sample_data[s_idx]["num_turns"]:
                    sample_data[s_idx]["num_turns"] = t_idx

        # Write to JSONL file
        with open(rollout_save_jsonl, "a", encoding="utf-8") as f:
            for s_idx in sorted(sample_data.keys()):
                data = sample_data[s_idx]
                first_idx = data["first_idx"]

                # Use multiturn_messages if available
                if multiturn_messages is not None and multiturn_messages[first_idx] is not None:
                    messages = multiturn_messages[first_idx]
                    entry = {
                        "messages": messages,
                        "score": data["total_score"],
                        "num_turns": data["num_turns"],
                    }
                else:
                    # Fallback to prompt + response
                    prompt = tokenizer.decode(
                        gen_batch_output.batch["prompts"][first_idx],
                        skip_special_tokens=True,
                    )
                    response = tokenizer.decode(
                        gen_batch_output.batch["responses"][first_idx],
                        skip_special_tokens=True,
                    )
                    entry = {
                        "conversation": prompt + response,
                        "score": data["total_score"],
                        "num_turns": data["num_turns"],
                    }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- 2. GENERATION ---
    print("Starting kernel code generation...")
    if async_rollout_mode:
        if wg is not None:
            dispatch_dp_size = wg.world_size
        elif async_rollout_manager is not None:
            dispatch_dp_size = async_rollout_manager.world_size
        else:
            raise RuntimeError("Async rollout mode enabled but no rollout manager is initialized")
    else:
        dispatch_dp_size = wg.world_size

    all_dataproto = []
    all_input_texts, all_output_texts = [], []
    progress_counter = 0

    # if dataproto_path and os.path.exists(dataproto_path):
        # read out the existing raw responses
        # print(f"Load DataProto from {dataproto_path}...")
        # all_dataproto_loaded = DataProto.load_from_disk(dataproto_path)
        
        # Process loaded data
        # print("Scoring loaded data...")
        # Treat as one large batch or split if needed. For simplicity, one batch.
        # process_batch(all_dataproto_loaded, batch_input_texts=None)
        
        # all_dataproto = all_dataproto_loaded
    # else:

    sample_inputs = []
    sample_outputs = []
    sample_scores = []
    async_rollout_diffs = []

    # Fix 1: UID-based conversation mapping for multi-turn alignment
    uid_to_conversation = {}  # Maps UID -> full conversation text

    for test_data in dataloader:

        print(f"Processing batch {progress_counter}...")
        test_batch = DataProto.from_single_dict(test_data)

        # CRITICAL: Convert uuid field FIRST before anything else
        # The vllm_async_engine reads "uuid" key (line 2169) and validates it as string
        # Must handle uuid BEFORE uid to ensure correct string conversion
        if "uuid" in test_batch.non_tensor_batch:
            # Convert existing uuid values to strings (handles integer/float types)
            existing_uuids = test_batch.non_tensor_batch["uuid"]
            if isinstance(existing_uuids, np.ndarray):
                test_batch.non_tensor_batch["uuid"] = np.array(
                    [str(u) for u in existing_uuids], dtype=object
                )
            elif isinstance(existing_uuids, list):
                test_batch.non_tensor_batch["uuid"] = np.array(
                    [str(u) for u in existing_uuids], dtype=object
                )
            else:
                # Handle scalar or other types
                test_batch.non_tensor_batch["uuid"] = np.array(
                    [str(existing_uuids)], dtype=object
                )

        # Add uid if not present (required for multi-turn tracking)
        # Ensure uid is always string type (convert if needed)
        # if "uid" not in test_batch.non_tensor_batch:
        #     test_batch.non_tensor_batch["uid"] = np.array(
        #         [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
        #     )
        # else:
            # Ensure existing uid values are strings (convert integers if needed)
            # existing_uids = test_batch.non_tensor_batch["uid"]
            # if isinstance(existing_uids, np.ndarray):
            #     # Convert all elements to strings
            #     test_batch.non_tensor_batch["uid"] = np.array(
            #         [str(uid) for uid in existing_uids], dtype=object
            #     )
            # elif isinstance(existing_uids, list):
            #     test_batch.non_tensor_batch["uid"] = np.array(
            #         [str(uid) for uid in existing_uids], dtype=object
            #     )

        # If uuid still doesn't exist after above checks, use uid as source
        # if "uuid" not in test_batch.non_tensor_batch:
            # Use uid as uuid source for consistency (uid is already string at this point)
            # test_batch.non_tensor_batch["uuid"] = test_batch.non_tensor_batch["uid"].copy()

        # repeat test batch
        test_batch = test_batch.repeat(
            repeat_times=config.data.n_samples, 
            interleave=True,
            )

        test_batch.non_tensor_batch["uid"] = np.array(
                [
                    f"test_example_{uuid4().hex}"
                    for i in range(len(test_batch.batch))
                ],
                dtype=object,
            )

        input_ids = test_batch.batch['input_ids']

        # Pop keys similar to _validate() in kernel_trainer
        # NOTE: Preserve multi-turn fields (turn_indices, uid, reward_extra_info) for metrics
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        # Don't pop turn_indices - keep it for multi-turn metrics

        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "uid"]
        # Don't pop uid - keep it for conversation grouping

        if "multi_modal_data" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")

        # if "reward_model" in test_batch.non_tensor_batch:
        #     non_tensor_batch_keys_to_pop.append("reward_model")
        # # Don't pop reward_extra_info if multi-turn - we need it for per-turn metrics
        # if not multi_turn_enabled and "reward_extra_info" in test_batch.non_tensor_batch:
        #     non_tensor_batch_keys_to_pop.append("reward_extra_info")
        # if "extra_info" in test_batch.non_tensor_batch:
        #     non_tensor_batch_keys_to_pop.append("extra_info")

        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        test_batch.non_tensor_batch["uid"] = test_gen_batch.non_tensor_batch["uid"]

        # CRITICAL: Copy uuid to test_gen_batch (engine always reads this at line 2169)
        # Must be done BEFORE any other operations
        # if 'uuid' in test_batch.non_tensor_batch:
        #     test_gen_batch.non_tensor_batch['uuid'] = test_batch.non_tensor_batch['uuid']
        #     # Debug: verify uuid types
        #     uuid_sample = test_gen_batch.non_tensor_batch['uuid'][0] if len(test_gen_batch.non_tensor_batch['uuid']) > 0 else None
        #     if uuid_sample is not None:
        #         print(f"DEBUG: uuid copied to test_gen_batch, type={type(uuid_sample)}, value={uuid_sample}")
        test_gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
                "global_step": 0,
            }
        # CRITICAL: Copy raw_prompt for async rollout mode (engine needs it at line 706)
        if (
            "reward_model" in test_batch.non_tensor_batch
            and async_rollout_mode
        ):
            test_gen_batch.non_tensor_batch[
                "reward_model"
            ] = test_batch.non_tensor_batch["reward_model"]
            test_gen_batch.non_tensor_batch[
                "data_source"
            ] = test_batch.non_tensor_batch["data_source"]

        # For multi-turn: Copy uid and other fields back to test_gen_batch
        # Also copy turn_indices if present
        # if multi_turn_enabled:
        #     if 'uid' in test_batch.non_tensor_batch:
        #         test_gen_batch.non_tensor_batch['uid'] = test_batch.non_tensor_batch['uid']
        #     if 'turn_indices' in test_batch.batch:
        #         test_gen_batch.batch['turn_indices'] = test_batch.batch['turn_indices']
        #     if 'data_source' in test_batch.non_tensor_batch:
        #         test_gen_batch.non_tensor_batch['data_source'] = test_batch.non_tensor_batch['data_source']
        #     if 'extra_info' in test_batch.non_tensor_batch:
        #         test_gen_batch.non_tensor_batch['extra_info'] = test_batch.non_tensor_batch['extra_info']

        # Debug: Verify test_gen_batch has required fields for async rollout
        if async_rollout_mode:
            print(f"[DEBUG] test_gen_batch for async rollout:")
            print(f"  - Batch size: {len(test_gen_batch)}")
            print(f"  - Has raw_prompt: {'raw_prompt' in test_gen_batch.non_tensor_batch}")
            print(f"  - Has raw_prompt_ids: {'raw_prompt_ids' in test_gen_batch.non_tensor_batch}")
            if 'raw_prompt' in test_gen_batch.non_tensor_batch:
                raw_prompts = test_gen_batch.non_tensor_batch['raw_prompt']
                print(f"  - raw_prompt count: {len(raw_prompts) if hasattr(raw_prompts, '__len__') else 'N/A'}")
                print(f"  - raw_prompt[0] type: {type(raw_prompts[0]) if len(raw_prompts) > 0 else 'empty'}")

        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, dispatch_dp_size)

        # Generate based on mode (sync or async)
        if not async_rollout_mode:
            test_output_gen_batch_padded = wg.generate_sequences(test_gen_batch_padded)
        else:
            async_rollout_manager.wake_up()
            test_output_gen_batch_padded = async_rollout_manager.generate_sequences(test_gen_batch_padded)
            async_rollout_manager.sleep()

        print('Generation complete for batch')

        if multi_turn_enabled:
            print(f"[DEBUG] After generation:")
            print(f"  - Output batch size: {len(test_output_gen_batch_padded)}")
            if 'turn_indices' in test_output_gen_batch_padded.batch:
                turn_indices = test_output_gen_batch_padded.batch['turn_indices'].cpu().numpy()
                print(f"  - Turn indices in output: {turn_indices}")
                print(f"  - Unique turns: {np.unique(turn_indices)}")
                # Count turns per sample
            
            if "uid" in test_output_gen_batch_padded.non_tensor_batch:
                uids = test_output_gen_batch_padded.non_tensor_batch["uid"]
                print(f"  - UIDs in output: {uids}")
                print(f"  - Unique UIDs: {np.unique(uids)}")

        # unpad
        if multi_turn_enabled:
            # Calculate actual max turns (accounts for multi-iteration)
            multi_iter_config = config.actor_rollout_ref.rollout.multi_turn.get('multi_iteration', {})
            if multi_iter_config.get('enable', False) and multi_iter_config.get('max_iterations', 1) > 1:
                # Multi-iteration: total turns = max_user_turns + (max_iterations - 1) * (max_user_turns - remain_turns)
                max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
                max_iterations = multi_iter_config.get('max_iterations', 1)
                remain_turns = multi_iter_config.get('remain_turns', 2)
                actual_max_turns = max_user_turns + (max_iterations - 1) * (max_user_turns - remain_turns)
            else:
                # Regular multi-turn: use max_user_turns
                actual_max_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns

            pad_size = pad_size * actual_max_turns
            print(f"[DEBUG] Multi-turn padding: pad_size multiplied by {actual_max_turns} turns")

        test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
        print("validation generation end")

        async_rollout_diffs.append(
                len(test_batch.batch) - len(test_output_gen_batch.batch)
            )

        # if batch is larger than gen_batch_output, which means that some prompts have been filtered out because of async timeout
        if len(test_batch.batch) > len(test_output_gen_batch.batch):
            # use uid to filter
            gen_uids = set(test_output_gen_batch.non_tensor_batch["uid"])
            batch_mask = np.array(
                [uid in gen_uids for uid in test_batch.non_tensor_batch["uid"]]
            )
            test_batch = test_batch[batch_mask]
            # guarantee uid alignment with generated outputs as in training
            assert False not in (
                test_batch.non_tensor_batch["uid"]
                == test_output_gen_batch.non_tensor_batch["uid"]
            )
            test_batch.non_tensor_batch[
                "uid"
            ] = test_output_gen_batch.non_tensor_batch["uid"]

        # Debug: Log output batch structure for multi-turn
        if multi_turn_enabled:
            print(f"[DEBUG] After generation:")
            print(f"  - Output batch size: {len(test_output_gen_batch)}")
            if 'turn_indices' in test_output_gen_batch.batch:
                turn_indices = test_output_gen_batch.batch['turn_indices'].cpu().numpy()
                print(f"  - Turn indices in output: {turn_indices}")
                print(f"  - Unique turns: {np.unique(turn_indices)}")
                # Count turns per sample
            
            if "uid" in test_output_gen_batch.non_tensor_batch:
                uids = test_output_gen_batch.non_tensor_batch["uid"]
                print(f"  - UIDs in output: {uids}")
                print(f"  - Unique UIDs: {np.unique(uids)}")

        if multi_turn_enabled:
            # For multi-turn, use multiturn_messages to build complete conversations
            sample_indices = (
                test_output_gen_batch.batch["sample_indices"].cpu().numpy()
            )

            # print(f"sample_indices before metrics: {sample_indices}")
            turn_indices = test_output_gen_batch.batch["turn_indices"].cpu().numpy()
            multiturn_messages = test_output_gen_batch.non_tensor_batch.get(
                "multiturn_messages", None
            )

            # Find first turn (with messages) for each sample
            sample_first_turn = {}  # sample_id -> row_idx of first turn
            for i in range(len(test_output_gen_batch.batch)):
                # s_idx = int(sample_indices[i])
                s_idx = test_output_gen_batch.non_tensor_batch["uid"][i]
                t_idx = int(turn_indices[i])

                if t_idx == -1:  # Skip padding turns
                    continue

                # First turn has the messages
                if s_idx not in sample_first_turn:
                    sample_first_turn[s_idx] = i

            # Build input/output for each sample using multiturn_messages
            for s_idx in sorted(sample_first_turn.keys()):
                first_idx = sample_first_turn[s_idx]

                if (
                    multiturn_messages is not None
                    and multiturn_messages[first_idx] is not None
                ):
                    messages = multiturn_messages[first_idx]

                    # Input: extract first user message
                    first_user_msg = ""
                    for msg in messages:
                        if msg.get("role") == "user":
                            first_user_msg = msg.get("content", "")
                            break
                    sample_inputs.append(first_user_msg)

                    # Output: build complete conversation string
                    full_output = ""
                    for msg in messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        full_output += f"[{role}]\n{content}\n\n"
                    sample_outputs.append(full_output)

                    # Fix 1: Store conversation by UID for alignment
                    uid_to_conversation[s_idx] = full_output
                else:
                    # Fallback to prompt/response decoding
                    first_prompt = tokenizer.decode(
                        test_output_gen_batch.batch["prompts"][first_idx],
                        skip_special_tokens=True,
                    )
                    sample_inputs.append(first_prompt)
                    fallback_msg = "[No messages available]"
                    sample_outputs.append(fallback_msg)

                    # Fix 1: Store fallback by UID
                    uid_to_conversation[s_idx] = fallback_msg
        else:
            # Original logic for single-turn
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

        if multi_turn_enabled:
            # Calculate actual max turns (accounts for multi-iteration)
            multi_iter_config = config.actor_rollout_ref.rollout.multi_turn.get('multi_iteration', {})
            if multi_iter_config.get('enable', False) and multi_iter_config.get('max_iterations', 1) > 1:
                # Multi-iteration: total turns = max_user_turns + (max_iterations - 1) * (max_user_turns - remain_turns)
                max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
                max_iterations = multi_iter_config.get('max_iterations', 1)
                remain_turns = multi_iter_config.get('remain_turns', 2)
                max_turns = max_user_turns + (max_iterations - 1) * (max_user_turns - remain_turns)
            else:
                # Regular multi-turn: use max_user_turns
                max_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns

            print(f"[DEBUG] Repeating test_batch by {max_turns} turns for multi-turn union")
            test_batch = test_batch.repeat(repeat_times=max_turns, interleave=True)

        test_batch = test_batch.union(test_output_gen_batch)

        # reward_tensor = test_batch.batch.pop("token_level_scores")
        reward_tensor = test_batch.batch["token_level_scores"]
        cur_data_source = test_batch.non_tensor_batch.get(
            "data_source", ["unknown"] * reward_tensor.shape[0]
        )
        reward_extra_info_raw = test_batch.non_tensor_batch.get(
            "reward_extra_info"
        )
        if reward_extra_info_raw is None:
            reward_extra_info_list = []
        elif hasattr(reward_extra_info_raw, "tolist"):
            reward_extra_info_list = reward_extra_info_raw.tolist()
        else:
            reward_extra_info_list = list(reward_extra_info_raw)

        valid_indices = []
        for i, d in enumerate(reward_extra_info_list):
            if len(d) > 0:
                # Check if dict has kernel-specific metrics (not just error info)
                has_kernel_metrics = any(
                    key in d
                    for key in [
                        "correctness",
                        "performance",
                        "compiled",
                        "success",
                    ]
                )
                if has_kernel_metrics:
                    valid_indices.append(i)

        valid_reward_extra_info_list = [
            reward_extra_info_list[i] for i in valid_indices
        ]
        valid_data_sources = [cur_data_source[i] for i in valid_indices]

        # convert list of dict to dict of list (only for valid entries with kernel metrics)
        if len(valid_reward_extra_info_list) > 0:
            raw_reward_extra_info_dict = {
                k: [d[k] for d in valid_reward_extra_info_list]
                for k in valid_reward_extra_info_list[0].keys()
            }

            if reward_extra_info_dict is None:
                reward_extra_info_dict = {}
            for key, extra_reward in raw_reward_extra_info_dict.items():
                for i, data_source in enumerate(valid_data_sources):
                    composed_key = f"{key}_{data_source}"
                    if composed_key not in reward_extra_info_dict:
                        reward_extra_info_dict[composed_key] = []
                    reward_extra_info_dict[composed_key].append(extra_reward[i])

        scores = reward_tensor.sum(-1).cpu().tolist()

        if multi_turn_enabled:
            # sample_indices = test_batch.batch['sample_indices'].cpu().numpy()
            sample_indices = test_batch.non_tensor_batch.get("uid", None)
            turn_indices = test_batch.batch["turn_indices"].cpu().numpy()

            # Aggregate scores by sample (sum of all turns)
            sample_score_map = {}
            for i in range(len(scores)):
                s_idx = sample_indices[i]
                t_idx = int(turn_indices[i])
                if t_idx == -1:  # Skip padding turns
                    continue
                if s_idx not in sample_score_map:
                    sample_score_map[s_idx] = 0.0
                sample_score_map[s_idx] += scores[i]

            # Add aggregated scores in the same UID order used for sample inputs/outputs above
            for s_idx in sorted(sample_score_map.keys()):
                sample_scores.append(sample_score_map[s_idx])
        else:
            sample_scores.extend(scores)


        # Log multi-turn conversations to JSONL during validation
        if (
            multi_turn_enabled
            and config.actor_rollout_ref.rollout.multi_turn.rollout_save_jsonl
            is not None
        ):
            _log_multiturn_to_jsonl(test_output_gen_batch, scores, config, tokenizer)

        reward_tensor_lst.append(reward_tensor)
        data_source_lst.append(cur_data_source)

        # Accumulate batch data for final processing
        all_dataproto.append(test_batch)

        # For raw response logging - decode all inputs/outputs
        input_ids_for_logging = test_batch.batch['input_ids']
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids_for_logging]
        all_input_texts.extend(input_texts)

        output_ids = test_output_gen_batch.batch['responses']
        output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        all_output_texts.extend(output_texts)

        progress_counter += 1

    all_dataproto = DataProto.concat(all_dataproto)

    if dataproto_path:
        # write the test batch into dataproto_path if we specified dataproto_path
        print(f"Saving DataProto to {dataproto_path}...")
        all_dataproto.save_to_disk(dataproto_path)

    if raw_response_path:
        samples = list(zip(all_input_texts, all_output_texts))
        # Group outputs by input text
        input_to_outputs = defaultdict(list)
        for input_text, output_text in samples:
            input_to_outputs[input_text].append(output_text)

        # Sort by input text for consistent ordering
        sorted_inputs = sorted(input_to_outputs.keys())

        # Write the output to raw_response_path as JSONL
        print(f"Saving raw responses to {raw_response_path}...")
        with open(raw_response_path, 'w') as f:
            for input_text in sorted_inputs:
                outputs = input_to_outputs[input_text]
                record = {"input": input_text, "outputs": outputs}
                # Write each record as a JSON line
                f.write(json.dumps(record, default=_json_default) + '\n')

    # Save artifacts (kernel.py, eval.json, etc.) for all samples
    print("Saving evaluation artifacts...")
    process_batch(all_dataproto, batch_input_texts=None)

    # Log global sample counter statistics (Fix 2)
    print(f"\nGlobal sample counter statistics:")
    for pid in sorted(global_sample_counter.keys()):
        print(f"  problem_{pid}: {global_sample_counter[pid]} samples")

    # Generate conversation summaries for multi-turn mode
    # Fix 1: Use UID-based mapping instead of lists for robust alignment
    print(f"Logging {len(sample_inputs)} multi-turn conversations...")
    if len(sample_inputs) > 0:
        print(f"Sample input: {sample_inputs[0]}")
        print(f"Sample output: {sample_outputs[0]}")
        print(f"Sample score: {sample_scores[0]}")
    print(f"UID-based conversation mapping contains {len(uid_to_conversation)} entries")
    generate_conversation_summaries(uid_to_conversation)

    # --- 3. SUMMARY AND FINAL METRICS ---
    print("Aggregating metrics...")

    # Concatenate all rewards
    print(f"Concatenating {len(reward_tensor_lst)} reward batches...")
    reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
    data_sources = np.concatenate(data_source_lst, axis=0)

    print(f"Total samples scored: {reward_tensor.shape[0]}")

    # --- 4. RICH METRICS COMPUTATION (like _validate) ---
    print("Computing rich metrics by data source...")

    # Group rewards by data source
    data_source_reward = {}
    for i in range(reward_tensor.shape[0]):
        data_source = data_sources[i]
        if data_source not in data_source_reward:
            data_source_reward[data_source] = []
        data_source_reward[data_source].append(reward_tensor[i].item())

    metric_dict = {}
    solve_threshold = config.data.get('solve_threshold', 0.99)
    k_for_pass_at_k = config.data.get('pass_at_k', min(config.data.n_samples, 1))

    for data_source, rewards in data_source_reward.items():
        # Mean reward
        metric_dict[f'val/test_score/{data_source}'] = float(np.mean(rewards))

        # Pass@k calculation (if n_samples > 1)
        if config.data.n_samples > 1 and len(rewards) % config.data.n_samples == 0:
            print(f"Calculating pass@{k_for_pass_at_k} rate for {data_source}...")
            reward_per_test_sample = np.reshape(rewards, (-1, config.data.n_samples))  # [N, n_samples]

            # Compute pass@k
            pass_at_k_rate = compute_pass_at_k(reward_per_test_sample, k=k_for_pass_at_k, threshold=solve_threshold)
            print(f"[{data_source}] pass@{k_for_pass_at_k} rate: {pass_at_k_rate:.4f}")
            metric_dict[f'val/test_score/{data_source}_pass@{k_for_pass_at_k}'] = pass_at_k_rate

    # Add reward_extra_info metrics - following kernel_trainer.py _validate() pattern
    if reward_extra_info_dict:
        print(f"Processing {len(reward_extra_info_dict)} extra metrics...")
        for key, extra_info_list in reward_extra_info_dict.items():
            # Normalize heterogeneous entries (dicts, tensors, scalars, etc.) to floats when possible
            coerced_values = [_coerce_extra_metric_value(v) for v in extra_info_list]
            valid_values = [v for v in coerced_values if v is not None]

            if valid_values:
                metric_dict[f'val/test_score_extra/{key}'] = float(np.mean(valid_values))
            else:
                metric_dict[f'val/test_score_extra/{key}'] = 0.0

            # Compute pass@k for score_ metrics (following trainer pattern)
            if not key.startswith('score_'):
                continue

            if not coerced_values or any(v is None for v in coerced_values):
                print(
                    f"Skipping pass@k computation for extra metric {key} due to missing/non-numeric values "
                    f"(total={len(coerced_values)})"
                )
                continue

            extra_rewards = list(coerced_values)

            n_val = config.data.n_samples
            k_val = k_for_pass_at_k

            if len(extra_rewards) % n_val != 0:
                print(f"Warning: extra validation samples for {key} not divisible by n, padding with 0.0 rewards")
                n_missing = n_val - (len(extra_rewards) % n_val)
                extra_rewards.extend([0.0] * n_missing)

            assert len(extra_rewards) % n_val == 0

            print(f"""Calculating pass@k rate for extra metric {key} with k={k_val}""")
            extra_rewards_per_sample = np.reshape(extra_rewards, (-1, n_val))
            extra_pass_at_k_rate = compute_pass_at_k(extra_rewards_per_sample, k=k_val, threshold=solve_threshold)
            print(f"[extra:{key}]pass_at_k_rate:", extra_pass_at_k_rate)
            metric_dict[f'val/test_score_extra/{key}_pass@{k_val}'] = extra_pass_at_k_rate

    # Compute multi-turn metrics if enabled
    if multi_turn_enabled and 'turn_indices' in all_dataproto.batch:
        print("=" * 80)
        print("Computing multi-turn metrics...")
        print("=" * 80)

        try:
            # General multi-turn metrics (turn counts, finish reasons, health)
            print("Computing general multi-turn metrics...")
            multi_turn_metrics = compute_multi_turn_metrics(all_dataproto)
            for key, val in multi_turn_metrics.items():
                metric_dict[f'val/{key}'] = val
                print(f"  {key}: {val}")
        except Exception as e:
            print(f"Warning: Failed to compute general multi-turn metrics: {e}")

        try:
            # Kernel-specific multi-turn metrics (per-turn performance, best-by-turn)
            print("Computing kernel-specific multi-turn metrics...")
            kernel_multi_turn_metrics = compute_kernel_multi_turn_metrics(
                all_dataproto,
                prefix="kernel"
            )
            for key, val in kernel_multi_turn_metrics.items():
                metric_dict[f'val/{key}'] = val

            # Print summary of key metrics
            print("\nKey Multi-Turn Metrics:")
            for key in sorted(kernel_multi_turn_metrics.keys()):
                if 'final' in key or 'improvement' in key or 'num_turns' in key:
                    print(f"  {key}: {kernel_multi_turn_metrics[key]}")
        except Exception as e:
            print(f"Warning: Failed to compute kernel multi-turn metrics: {e}")

        print("=" * 80)

    # --- 5. SAVING RESULTS ---
    print("Saving results to JSONL...")

    # Group scores by prompt_index for solve_rate calculation
    prompt_indices = all_dataproto.non_tensor_batch['prompt_index']
    grouped_scores = defaultdict(list)

    score_results = reward_tensor.tolist()
    for i, prompt_idx in enumerate(prompt_indices):
        grouped_scores[prompt_idx].append(score_results[i])

    # Calculate solve rate for each prompt
    solve_rates = {}
    for prompt_idx, scores in grouped_scores.items():
        scores_array = np.array(scores)
        solve_count = np.sum(scores_array >= solve_threshold)
        total_count = len(scores_array)
        solve_rate = solve_count / total_count if total_count > 0 else 0.0
        solve_rates[prompt_idx] = solve_rate

    # Add the calculated 'solve_rate' to the DataFrame
    solve_rates_series = pd.Series(solve_rates, name='solve_rate')
    dataframe = dataset.dataframe
    dataframe['solve_rate'] = dataframe.index.map(solve_rates_series)
    # filter where the solve_rate is not NaN
    dataframe = dataframe[dataframe['solve_rate'].notna()]

    # Save/append the final dataset to a JSONL file
    output_path = config.data.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = 'a' if os.path.exists(output_path) else 'w'
    with open(output_path, mode) as f:
        for record in dataframe.to_dict(orient='records'):
            f.write(json.dumps(record, default=_json_default) + '\n')

    # Save conversations to JSONL if multi-turn
    save_conversations_to_jsonl(all_dataproto, reward_tensor, tokenizer, output_path)

    # --- 6. PRINT SUMMARY ---
    avg_solve_rate = dataframe['solve_rate'].mean()
    print(f"\n{'='*80}")
    print(f"Kernel Code Grading Summary")
    print(f"{'='*80}")
    print(f"Total prompts evaluated: {len(dataframe)}")
    print(f"Average solve rate: {avg_solve_rate:.2%}")
    print(f"Solve threshold: {solve_threshold}")
    print(f"\nMetrics by data source:")
    for key, value in sorted(metric_dict.items()):
        if 'test_score/' in key and not 'pass@' in key:
            print(f"  {key}: {value:.4f}")
    print(f"\nPass@k metrics:")
    for key, value in sorted(metric_dict.items()):
        if 'pass@' in key:
            print(f"  {key}: {value:.4f}")

    if reward_extra_info_dict:
        print(f"\nExtra kernel metrics (top 10):")
        extra_metrics = {k: v for k, v in metric_dict.items() if 'test_score_extra/' in k}
        for i, (key, value) in enumerate(sorted(extra_metrics.items())[:10]):
            metric_name = key.replace('val/test_score_extra/', '')
            print(f"  {metric_name}: {value:.4f}")
        if len(extra_metrics) > 10:
            print(f"  ... and {len(extra_metrics) - 10} more metrics")

    print(f"\nOutput saved to: {config.data.output_path}")
    print(f"{'='*80}\n")

    # Log to wandb if enabled
    if use_wandb:
        try:
            import wandb

            # Log all metrics
            wandb.log(metric_dict, step=0)
            print("✅ Metrics logged to wandb")

            # Log sample generations using collected inputs/outputs/scores
            # (same pattern as kernel_trainer.py _maybe_log_val_generations line 2064-2082)
            generations_to_log = config.trainer.get('log_val_generations', 10)
            if generations_to_log > 0 and sample_inputs and sample_outputs and sample_scores:
                samples = list(zip(sample_inputs, sample_outputs, sample_scores))
                samples.sort(key=lambda x: x[0])  # Sort by input text

                rng = np.random.RandomState(42)
                rng.shuffle(samples)

                samples = samples[:generations_to_log]

                validation_logger.log(
                    config.trainer.logger, samples, step=0
                )
                print(f"✅ Logged {len(samples)} sample generations to wandb")

        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")

    # Optionally save metrics to JSON
    metrics_path = config.data.get('metrics_output_path')
    if metrics_path:
        with open(metrics_path, 'w') as f:
            json.dump(metric_dict, f, indent=2, default=_json_default)
        print(f"Metrics saved to: {metrics_path}")

    # Finish wandb run
    if use_wandb:
        try:
            import wandb
            wandb.finish()
            print("✅ Wandb run finished")
        except Exception as e:
            print(f"Warning: Failed to finish wandb: {e}")

    return eval_output_dir


if __name__ == '__main__':
    main()
