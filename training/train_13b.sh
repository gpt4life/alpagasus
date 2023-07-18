DATA_DIR=<Your DATA PATH>
OUTPUT_DIR=<OUTPUT DIRECTORY>

# suggest: elinas/llama-7b-hf-transformers-4.29

torchrun --nproc_per_node=8 --master_port=54321 train_alpaca.py \
    --model_name_or_path /path/to/llama-13b \
    --data_path ${DATA_DIR} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --report_to "none" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True