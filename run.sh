DATASET_NAME="ICFG-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name LPNC \
--output_dir 'LPNC_log' \
--dataset_name $DATASET_NAME \
--loss_names 'supid+cotrl+cid' \
--num_epoch 60
