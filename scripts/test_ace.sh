#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../")

DATASET_PATH_TEST=$(realpath -s "${REPO_PATH}/datasets/7scenes_*")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

############################################### Train Test ACE Heads #######################################
out_dir="${REPO_PATH}/logs/mapfree/test"
mkdir -p "$out_dir"

for scene in ${DATASET_PATH_TEST}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  python $testing_exe "${scene}" "/workspace/project/marepo/logs/pretrain/ace_models/7Scenes/7scenes_testBL.pt" --test_batch_size 64 --save_result False\
  2>&1 | tee "$out_dir/${scene##*/}/log_${scene##*/}.txt"
done

############################################### Train Test ACE Heads #######################################
# # Test on different batch size
# test_batch_sizes=(1 8 32 64)
# out_dir="${REPO_PATH}/logs/mapfree/test"
# mkdir -p "$out_dir"

# for scene in ${DATASET_PATH_TEST}; do
#   echo "${scene}" # whole path
#   echo "${scene##*/}" # base file name
#   for test_batch_size in "${test_batch_sizes[@]}"; do
#     echo "Test_batch_size: ${test_batch_size}"
#     python $testing_exe "${scene}" "/workspace/project/marepo/logs/pretrain/ace_models/7Scenes/7scenes_testBL.pt" --test_batch_size ${test_batch_size} --save_result True --session "b${test_batch_size}"\
#   2>&1 | tee "$out_dir/${scene##*/}/log_${scene##*/}.txt"
#   done
# done