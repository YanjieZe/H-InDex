# bash scripts/adroit/gen_img_dataset.sh hammer
# bash scripts/adroit/gen_img_dataset.sh door
# bash scripts/adroit/gen_img_dataset.sh pen

# hammer, door, pen
gpu=0

if [[ $1 == "hammer" ]]; then
  camera_name="top"
elif [[ $1 == "door" ]]; then
  camera_name="top"
elif [[ $1 == "pen" ]]; then
  camera_name="vil_camera"
elif [[ $1 == "relocate" ]]; then
  camera_name="vil_camera"
fi


env_name="${1}-v0"
data_dir="../../AdroitImgDataset/${env_name}/"
cd stage3_RL/examples
CUDA_VISIBLE_DEVICES=${gpu} python createImgData_adroit.py --data_dir ${data_dir} \
                                --env_name ${env_name} \
                                --num_demos 50 \
                                --mode exploration \
                                --policy ../../third_party/hand_dapg/dapg/policies/${env_name}.pickle \
                                --camera_name ${camera_name}                              