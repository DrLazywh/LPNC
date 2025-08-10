DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name PPL \
--output_dir 'PPL_logs1' \
--dataset_name $DATASET_NAME \
--loss_names 'supid+cotrl+cid' \
--num_epoch 60 

