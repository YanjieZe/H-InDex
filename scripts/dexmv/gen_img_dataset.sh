# bash scripts/dexmv/gen_img_dataset.sh pour
# bash scripts/dexmv/gen_img_dataset.sh place_inside
# bash scripts/dexmv/gen_img_dataset.sh relocate-mug
# bash scripts/dexmv/gen_img_dataset.sh relocate-foam_brick
# bash scripts/dexmv/gen_img_dataset.sh relocate-large_clamp
# bash scripts/dexmv/gen_img_dataset.sh relocate-mustard_bottle
# bash scripts/dexmv/gen_img_dataset.sh relocate-potted_meat_can
# bash scripts/dexmv/gen_img_dataset.sh relocate-sugar_box
# bash scripts/dexmv/gen_img_dataset.sh relocate-tomato_soup_can

# task name: 
#   pour, place_inside
#   relocate-mug, relocate-foam_brick
#   relocate-large_clamp, relocate-mustard_bottle
#   relocate-potted_meat_can, relocate-sugar_box
#   relocate-tomato_soup_can

gpu=6
num_demos=50
# frontview, sideview, backview
# cam=sideview
cam=backview
# cam=frontview

cd stage3_RL/examples
task_name=${1}
data_dir="../../AdroitImgDataset/${task_name}/"
CUDA_VISIBLE_DEVICES=${gpu} python createImgData_dexmv.py --data_dir ${data_dir} \
                                --env_name ${task_name} \
                                --num_demos ${num_demos} \
                                --cam ${cam}
