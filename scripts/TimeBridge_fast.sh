#!/bin/bash

if [ ! -d "./logs/LongForecasting/TimeBridge" ]; then
    mkdir -p ./logs/LongForecasting/TimeBridge
fi

model_name=TimeBridge
seq_len=720
pred_len=96        # smallest horizon in the paper
GPU=0
root=./dataset

###########################
#      ETTh1  (10 epochs)
###########################
alpha=0.35
data_name=ETTh1

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/ETT-small/ \
  --data_path $data_name.csv \
  --model_id ${data_name}_${seq_len}_${pred_len}_ep10 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
  --ca_layers 0 \
  --pd_layers 1 \
  --ia_layers 3 \
  --des 'Exp_ep10' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 64 \
  --alpha $alpha \
  --learning_rate 0.0002 \
  --train_epochs 10 \
  --patience 5 \
  --itr 1 > logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_ep10.logs


###########################
#      ETTh2  (10 epochs)
###########################
alpha=0.35
data_name=ETTh2

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/ETT-small/ \
  --data_path $data_name.csv \
  --model_id ${data_name}_${seq_len}_${pred_len}_ep10 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
  --period 48 \
  --ca_layers 0 \
  --pd_layers 1 \
  --ia_layers 3 \
  --des 'Exp_ep10' \
  --n_heads 4 \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 5 \
  --alpha $alpha \
  --batch_size 16 \
  --itr 1 > logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_ep10.logs


###########################
#      ETTm1  (10 epochs)
###########################
alpha=0.35
data_name=ETTm1

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/ETT-small/ \
  --data_path $data_name.csv \
  --model_id ${data_name}_${seq_len}_${pred_len}_ep10 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
  --ca_layers 0 \
  --pd_layers 1 \
  --ia_layers 3 \
  --des 'Exp_ep10' \
  --n_heads 4 \
  --d_model 64 \
  --d_ff 128 \
  --period 48 \
  --num_p 6 \
  --lradj 'TST' \
  --learning_rate 0.0002 \
  --train_epochs 10 \
  --pct_start 0.2 \
  --patience 5 \
  --batch_size 64 \
  --alpha $alpha \
  --itr 1 > logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_ep10.logs


###########################
#      ETTm2  (10 epochs)
###########################
alpha=0.35
data_name=ETTm2

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/ETT-small/ \
  --data_path $data_name.csv \
  --model_id ${data_name}_${seq_len}_${pred_len}_ep10 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
  --ca_layers 0 \
  --pd_layers 1 \
  --ia_layers 3 \
  --des 'Exp_ep10' \
  --n_heads 4 \
  --d_model 64 \
  --d_ff 128 \
  --period 48 \
  --lradj 'TST' \
  --learning_rate 0.0002 \
  --pct_start 0.2 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --alpha $alpha \
  --itr 1 > logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_ep10.logs


###########################
#      Weather  (10 epochs)
###########################
alpha=0.10
data_name=weather

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/weather/ \
  --data_path weather.csv \
  --model_id ${data_name}_${seq_len}_${pred_len}_ep10 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 21 \
  --ca_layers 1 \
  --pd_layers 1 \
  --ia_layers 1 \
  --des 'Exp_ep10' \
  --period 48 \
  --num_p 12 \
  --d_model 128 \
  --d_ff 128 \
  --alpha $alpha \
  --learning_rate 0.0002 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --itr 1 > logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_ep10.logs


###########################
#      Traffic  (10 epochs)
###########################
alpha=0.35
data_name=traffic

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/traffic/ \
  --data_path traffic.csv \
  --model_id ${data_name}_${seq_len}_${pred_len}_ep10 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 862 \
  --des 'Exp_ep10' \
  --num_p 8 \
  --n_heads 64 \
  --stable_len 2 \
  --d_ff 512 \
  --d_model 512 \
  --ca_layers 3 \
  --pd_layers 1 \
  --ia_layers 1 \
  --batch_size 4 \
  --attn_dropout 0.15 \
  --patience 5 \
  --train_epochs 10 \
  --learning_rate 0.0005 \
  --alpha $alpha \
  --itr 1 > logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_ep10.logs
