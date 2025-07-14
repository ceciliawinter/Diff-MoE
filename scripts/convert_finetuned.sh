#for finetuned model, the finetuned adapter model should be saved in /data and named as $model-$dataset
for model in  switch-base-128 switch-large-128 do
for dataset in squad xsum coqa
    do
        echo $model
        rm -rf /data/ft/$model
        python /workspace/FasterTransformer/examples/pytorch/t5/utils/huggingface_switch_transformer_ckpt_convert.py -saved_dir /data/ft/$model-$dataset -in_file google/$model -inference_tensor_para_size 1 -use_base_model True -use_base_model_path /data/$model-$dataset 
        mv /data/ft/$model/1-gpu/* /data/ft/$model/
        rm -d /data/ft/$model/1-gpu
    done
done
 