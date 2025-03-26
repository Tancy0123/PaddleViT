CUDA_VISIBLE_DEVICES=0 \
python main_multi_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-batch_size_eval=256 \
-data_path='/data1/dataset/classification/ISLVRC2012/ImageNet' \
-eval \
-amp
