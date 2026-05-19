#!/bin/bash

model_name=$1
save_name=$2

pids=()

for i in {0..7}; do
  env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u RAY_LOCAL_RANK -u RAY_LOCAL_WORLD_SIZE \
    CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate.py --model $model_name --suffix $i --save_name $save_name &
  pids[$i]=$!
  sleep 1
done

for i in {0..7}; do
  wait ${pids[$i]} 2>/dev/null
done
