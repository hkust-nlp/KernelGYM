
import hydra
from verl_patch.trainer.code.fsdp_sft_trainer import create_sft_dataset, FSDPSFTTrainer
from verl.utils.distributed import initialize_global_process_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .constant import QWEN3CHATTEMPLATE

from verl.utils.fs import copy_to_local

@hydra.main(config_path="../verl_patch/trainer/code/config", config_name="sft_trainer", version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp")
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    if "qwen3" in local_model_path.lower() or "qwen-3" in local_model_path.lower():
        # fix qwen3 chat template
        if "coder" not in local_model_path.lower():
            tokenizer.chat_template = QWEN3CHATTEMPLATE

    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

if __name__ == "__main__":
    main()