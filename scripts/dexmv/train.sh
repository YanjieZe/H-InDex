# bash scripts/dexmv/train.sh pour hindex test 0 0

use_wandb=1
num_cpu=8
# task name: 
#   pour, place_inside, relocate-mug, relocate-foam_brick
#   relocate-large_clamp, relocate-mustard_bottle
#   relocate-potted_meat_can, relocate-sugar_box
#   relocate-tomato_soup_can

task=${1}
# available encoder type: 
#   hindex, vc1, mvp, r3m, rrl

encoder_type=${2}
addition_info=${3}
seed=${4}
gpu=${5}
wandb_group="${task}-${encoder_type}-${addition_info}"


cur_dir=$(pwd)

if [[ $encoder_type == "hindex" ]]; then
  encoder_ckpt="${cur_dir}/archive/adapted_frankmocap_hand_ckpts/${task}.pth"
elif [[ $encoder_type == "r3m" ]]; then
  encoder_ckpt="${cur_dir}/archive/r3m_ckpt"
elif [[ $encoder_type == "mvp" ]]; then
  encoder_ckpt="${cur_dir}/archive/mae_pretrain_hoi_vit_small.pth"
elif [[ $encoder_type == "vc1" ]]; then
  encoder_ckpt="${cur_dir}/archive/vc1_vitb.pth"
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
            output=${wandb_group} \
            encoder_ckpt=${encoder_ckpt} \
            encoder_type=${encoder_type} \
            --config-name ${config} \
                


