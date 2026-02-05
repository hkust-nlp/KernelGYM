# bash drkernel/kernel/scripts/preprocess/push_to_hub.sh "hkust-nlp/drkernel-coldstart-8k" data/cold-start-data "dataset"
# bash drkernel/kernel/scripts/preprocess/push_to_hub.sh "hkust-nlp/drkernel-rl" data/cold-start-data "dataset"
# bash drkernel/kernel/scripts/preprocess/push_to_hub.sh "hkust-nlp/drkernel-14b" /mnt/hdfs/weiliu/kernel-ckpts/0114-bsz16-14b-filterref-mt3-speedup-trloo-tcoverage0.5-turnrs-geomis-036rc_e3b0c442_cudallm-data_processed_cuda_llm_rl_thinking_1025_MTv2-VEP-GPT5M-fixop-qwen3-14b-base_Cudallm-MTv2-VEP-GPT5M-Fixop_qwen3-14b-base/global_step_556/global_step_230/actor/huggingface "model"
bash drkernel/kernel/scripts/preprocess/push_to_hub.sh "hkust-nlp/drkernel-validation-data" data/val-data/ "dataset"
