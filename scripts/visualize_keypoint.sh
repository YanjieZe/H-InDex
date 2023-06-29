# demo:
# bash scripts/visualize_keypoint.sh door-v0

task_name=${1}

# task name: 
#   door-v0, hammer-v0, pen-v0
#   pour, place_inside
#   relocate-mug, relocate-foam_brick
#   relocate-large_clamp, relocate-mustard_bottle
#   relocate-potted_meat_can, relocate-sugar_box
#   relocate-tomato_soup_can


use_entire_pretrain=1 
freeze_encoder=0
freeze_bn=0



exp_name="frankmocap_hand"
config="ttp"



# origin
# MODEL_FILE="penn_joint.pth"

cur_dir=$(pwd)
MODEL_FILE="${cur_dir}/archive/adapted_frankmocap_hand_ckpts/${task_name}.pth"
keypointnet_pretrain="none"
data_root="${cur_dir}/AdroitImgDataset"



gpu=0
cd stage2_adapt
CUDA_VISIBLE_DEVICES=${gpu} python tools/visualize_keypoint.py \
      --cfg experiments/${config}.yaml \
      --use_wandb 0 \
      --wandb_group ${exp_name} \
      --use_entire_pretrain ${use_entire_pretrain} \
      --freeze_encoder ${freeze_encoder} \
      --resume 0 \
      --keypointnet_pretrain ${keypointnet_pretrain} \
      --freeze_bn ${freeze_bn} \
      --task_name ${task_name} \
      TEST.MODEL_FILE ${MODEL_FILE} \
      DATASET.ROOT ${data_root}