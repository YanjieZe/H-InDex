# bash scripts/dexmv/train.sh place_inside ttp_frankmocap_hand_bn0.01 withSR 0 0
# bash scripts/dexmv/train.sh relocate-potted_meat_can mvp withSR 0 0


use_wandb=1
num_cpu=8
# task name: 
#   pour, place_inside
#   relocate-mug, 
#   relocate-foam_brick
#   relocate-large_clamp, relocate-mustard_bottle
#   relocate-potted_meat_can, relocate-sugar_box
#   relocate-tomato_soup_can

task=${1}
# available encoder type: 
#   resnet50/resnet34, rrl, dino, r3m, pvr, mvp, 
#   frankmocap_hand, frankmocap_body, alphapose,
#   hand_object_detector, vc1

# current sota:
#   ttp_frankmocap_hand_bn, resnet50_bn

encoder_type=${2}
use_encoder_adaptation=0
addition_info=${3}
seed=${4}
gpu=${5}
wandb_group="${task}-${encoder_type}-${addition_info}"


test_time_momentum=0.
if [[ $encoder_type == "ttp_frankmocap_hand_bn"* || $encoder_type == "frankmocap_hand_bn"* || $encoder_type == "resnet50_bn"* ]]; then
  test_time_momentum="${encoder_type##*_bn}"
  encoder_type="${encoder_type%_*}_bn"
else
  test_time_momentum=0.
fi


cur_dir=$(pwd)


if [[ $encoder_type == "frankmocap_hand" ]]; then
  encoder_ckpt="${cur_dir}/archive/frankmocap_hand.pth"
elif [[ $encoder_type == "frankmocap_hand_bn" ]]; then
  encoder_ckpt="${cur_dir}/archive/frankmocap_hand.pth"
elif [[ $encoder_type == "frankmocap_hand_fusion" ]]; then
  encoder_ckpt="${cur_dir}/archive/frankmocap_hand.pth"
elif [[ $encoder_type == "frankmocap_hand_joint" ]]; then
  encoder_ckpt="${cur_dir}/archive/frankmocap_hand.pth"
elif [[ $encoder_type == "frankmocap_hand_state" ]]; then
  encoder_ckpt="${cur_dir}/archive/frankmocap_hand.pth"
elif [[ $encoder_type == "frankmocap_body" ]]; then
  encoder_ckpt="${cur_dir}/archive/frankmocap_body.pt"
elif [[ $encoder_type == "dino" ]]; then
  encoder_ckpt="${cur_dir}/archive/dino_deitsmall8_pretrain.pth"
elif [[ $encoder_type == "r3m" ]]; then
  encoder_ckpt="${cur_dir}/archive/r3m_ckpt"
elif [[ $encoder_type == "pvr" ]]; then
  encoder_ckpt="${cur_dir}/archive/moco_v2_800ep_pretrain.pth.tar"
elif [[ $encoder_type == "mvp" ]]; then
  encoder_ckpt="${cur_dir}/archive/mae_pretrain_hoi_vit_small.pth"
elif [[ $encoder_type == "vc1" ]]; then
  encoder_ckpt="${cur_dir}/archive/vc1_vitb.pth"
elif [[ $encoder_type == "alphapose" ]]; then
  encoder_ckpt="${cur_dir}/archive/fast_res50_256x192.pth"
elif [[ $encoder_type == "hand_object_detector" ]]; then
  encoder_ckpt="${cur_dir}/archive/faster_rcnn_1_8_132028.pth"
elif [[ $encoder_type == "ttp_frankmocap_hand" ]]; then
  encoder_ckpt="${cur_dir}/archive/adapted_frankmocap_hand_ckpts/${task}.pth"
elif [[ $encoder_type == "ttp_frankmocap_hand_bn" ]]; then
  encoder_ckpt="${cur_dir}/archive/adapted_frankmocap_hand_ckpts/${task}.pth"
elif [[ $encoder_type == "ttp_human" ]]; then
  encoder_ckpt="${cur_dir}/archive/penn_joint.pth"
else
  encoder_ckpt="none"
fi

config="${task}_dapg"

cd stage3_RL/examples


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin

CUDA_VISIBLE_DEVICES=${gpu} python train_dexmv.py \
            use_wandb=${use_wandb} \
            wandb_project="hand_rl" \
            wandb_group=${wandb_group} \
            wandb_name=${seed} \
            num_cpu=${num_cpu} \
            seed=${seed} \
            use_bc=1 \
            hybrid_state=1 \
            output=${wandb_group} \
            use_encoder_adaptation=${use_encoder_adaptation} \
            test_time_momentum=${test_time_momentum} \
            encoder_ckpt=${encoder_ckpt} \
            encoder_type=${encoder_type} \
            --config-name ${config} \
                


