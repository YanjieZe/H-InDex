# demo:
#   bash scripts/train_stage2.sh relocate-sugar_box

# task name: 
#   door-v0, hammer-v0, pen-v0
#   pour, place_inside
#   relocate-mug, relocate-foam_brick
#   relocate-large_clamp, relocate-mustard_bottle
#   relocate-potted_meat_can, relocate-sugar_box
#   relocate-tomato_soup_can

task_name=$1
gpu=0

learning_rate=0.0001
train_iter=100000

freeze_conv=1
freeze_bn=0
exp_name="frankmocap_hand_${task_name}_fc${freeze_conv}_fb${freeze_bn}"
config="ttp"


# origin
MODEL_FILE="penn_joint.pth"
cur_dir=$(pwd)
keypointnet_pretrain="${cur_dir}/archive/frankmocap_hand.pth"
data_root="${cur_dir}/AdroitImgDataset"

use_entire_pretrain=1

freeze_encoder=0




cd stage2_adapt

start_time=$(date +%s)

CUDA_VISIBLE_DEVICES=${gpu} python tools/test_time_training.py \
      --cfg experiments/${config}.yaml \
      --use_wandb 0 \
      --wandb_group ${exp_name} \
      --use_entire_pretrain ${use_entire_pretrain} \
      --freeze_encoder ${freeze_encoder} \
      --resume 0 \
      --keypointnet_pretrain ${keypointnet_pretrain} \
      --freeze_bn ${freeze_bn} \
      --freeze_conv ${freeze_conv} \
      --task_name ${task_name} \
      TEST.MODEL_FILE ${MODEL_FILE} \
      TRAIN.NUM_ITERS ${train_iter} \
      TRAIN.EVAL_FREQ 500 \
      TRAIN.LR ${learning_rate} \
      DATASET.ROOT ${data_root}


end_time=$(date +%s)
cost_time=$((end_time - start_time))
cost_time_hours=$(echo "scale=4; $cost_time / 3600" | bc)
# print time cost using hours
echo -e "\033[36mexp name: ${exp_name}\033[0m"
echo -e "\033[36mstart time: $(date -d @${start_time} +"%Y-%m-%d %H:%M:%S")\033[0m"
echo -e "\033[36mend time: $(date -d @${end_time} +"%Y-%m-%d %H:%M:%S")\033[0m"
echo -e "\033[36mcost time: ${cost_time_hours} hours\033[0m"

