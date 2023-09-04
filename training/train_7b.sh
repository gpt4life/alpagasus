DATA_DIR=../rating/alpaca_filtered_data.json
MODEL_DIR=../models/llama-7b
OUTPUT_DIR=../models/alpagasus-7b

torchrun --nproc_per_node=4 --master_port=54321 train_alpaca.py \
    --model_name_or_path ${MODEL_DIR} \
    --data_path ${DATA_DIR} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --report_to "none" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True