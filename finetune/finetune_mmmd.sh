#! /bin/bash

#adapted from finetune_visualglm.sh
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="visualglm-6b"
MODEL_ARGS="--max_source_length 1760 \
    --max_target_length 256 \
    --lora_rank 10 \
    --layer_range 0 14 \
    --pre_seq_len 4"

# OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"
OPTIONS_CUDA="CUDA_VISIBLE_DEVICES=0"

train_data="/home/bchen/projects/mmmd/data/data/train.jsonl"
eval_data="/home/bchen/projects/mmmd/data/data/valid.jsonl"
img_base="/home/bchen/projects/mmmd/data/imgs"

# --train-iters 300 \
gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --resume-dataloader \
       --epochs 5 \
       $MODEL_ARGS \
       --img_base_dir ${img_base} \
       --train-data ${train_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 2000 \
       --eval-interval 2000 \
       --save "./checkpoints" \
       --split 1 \
       --eval-iters 10 \
       --eval-batch-size 8 \
       --zero-stage 1 \
       --lr 0.0001 \
       --batch-size 2 \
       --skip-init \
       --fp16 \
       --use_lora
"

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} ${OPTIONS_CUDA} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_visualglm_mmmd.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
