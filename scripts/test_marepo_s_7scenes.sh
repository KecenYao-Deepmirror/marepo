#!/bin/bash
### Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../")

model_name="marepo" # Paper Model
out_dir="${REPO_PATH}/logs/${model_name}"
mkdir "${out_dir}"
############ example to finetuned marepo_s ###############

########### benchmark on Finetuned MarepoS ###############
### TEST
testing_exe="${REPO_PATH}/test_marepo.py"
read_log_Marepo="${REPO_PATH}/read_log_marepo.py" # for computing scene average stats
DATASET_PATH_TEST=$(realpath -s "${REPO_PATH}/datasets/7scenes_*")

datatype="test"
for scene in ${DATASET_PATH_TEST}; do
  if [ ${scene##*/} = "7scenes_source" ]
  then
    echo "skip 7scenes_source".
  else
    echo "${scene}" # whole path
    echo "${scene##*/}" # base file name
    ace_head_path="${REPO_PATH}/logs/pretrain/ace_models/7Scenes/${scene##*/}.pt"
    marepo_head_path="${REPO_PATH}/logs/marepo_${scene##*/}_240405/marepo_${scene##*/}_240405-600ep.pt"
    OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,1 python $testing_exe "${scene}" "$marepo_head_path" --head_network_path ${ace_head_path} \
    --transformer_json ../transformer/config/nerf_focal_12T1R_256_homo.json --load_scheme2_sc_map True --save_result True --test_batch_size 1\
    2>&1 | tee "$out_dir/log_Finetune_Marepo_${scene##*/}_${datatype}.txt"
  fi
done
python $read_log_Marepo "7Scenes" "${out_dir}" "${datatype}" --finetune True



# # Test on different batch size
# test_batch_sizes=(64)
# testing_exe="${REPO_PATH}/test_marepo.py"
# read_log_Marepo="${REPO_PATH}/read_log_marepo.py" # for computing scene average stats
# DATASET_PATH_TEST=$(realpath -s "${REPO_PATH}/datasets/7scenes_*")

# datatype="test"
# for scene in ${DATASET_PATH_TEST}; do
#   if [ ${scene##*/} = "7scenes_source" ]
#   then
#     echo "skip 7scenes_source".
#   else
#     echo "${scene}" # whole path
#     echo "${scene##*/}" # base file name
#     ace_head_path="${REPO_PATH}/logs/pretrain/ace_models/7Scenes/${scene##*/}.pt"
#     marepo_head_path="${REPO_PATH}/logs/marepo_${scene##*/}_240405/marepo_${scene##*/}_240405-600ep.pt"
#     for test_batch_size in "${test_batch_sizes[@]}"; do
#       echo "Test_batch_size: ${test_batch_size}"
#       OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,1 python $testing_exe "${scene}" "$marepo_head_path" --head_network_path ${ace_head_path} \
#       --transformer_json ../transformer/config/nerf_focal_12T1R_256_homo.json --load_scheme2_sc_map True --test_batch_size ${test_batch_size} --save_result False --session "b${test_batch_size}"\
#       2>&1 | tee "$out_dir/log_Finetune_Marepo_${scene##*/}_${datatype}_b${test_batch_size}.txt"
#     done
#   fi
# done