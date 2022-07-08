#!bin/sh

python inference.py \
--ckpt_path /home/epoch=82-val_loss=0.7781.ckpt \
--input_dir /home/dataset \
--output_dir /home/output_dir \
--save-vis