export CUDA_VISIBLE_DEVICES=0
cd ~/AlpaGasus/generation
for training_data_source_name in 'alpaca' # your model name here
do
    model_name=${training_data_source_name}
    mkdir -p ~/results/${model_name}/${model_name}

    for dataset_name in 'koala' 'vicuna' 'koala' 'wizardlm'
    do
        python generation_v2.py \
            --model_name_or_path /Path/to/your/model \
            --training_data_source_name ${training_data_source_name} \
            --dataset_name ${dataset_name}
    done
done