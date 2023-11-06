<p align="center">

  <h1 align="center"><a href="https://yanjieze.com/H-InDex/">H-InDex</a>:
Visual Reinforcement Learning with <a href="https://yanjieze.com/H-InDex/">H</a>and<a href="https://yanjieze.com/H-InDex/">-In</a>formed Representations for <a href="https://yanjieze.com/H-InDex/">Dex</a>terous Manipulation</h1>
<h2 align="center">NeurIPS 2023</h2>
  <p align="center">
    <a><strong>Yanjie Ze</strong></a>
    ·
    <a><strong>Yuyao Liu*</strong></a>
    ·
    <a><strong>Ruizhe Shi*</strong></a>
    ·
    <a><strong>Jiaxin Qin</strong></a>
    ·
    <a><strong>Zhecheng Yuan</strong></a>
    ·
    <a><strong>Jiashun Wang</strong></a>
    ·
    <a><strong>Xiaolong Wang</strong></a>
    ·
    <a><strong>Huazhe Xu</strong></a>
  </p>

</p>

<h3 align="center">
  <a href="https://yanjieze.com/H-InDex/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2310.01404"><strong>arXiv</strong></a>
  |
  <a href=""><strong>Twitter</strong></a>
</h3>

<div align="center">
  <img src="teaser.png" alt="Logo" width="100%">
</div>

# 🧾 Introduction
H-InDex is a visual reinforcement learning framework that leverages hand-informed representations to learn dexterous manipulation skills with great efficiency. H-InDex consistes of three stages: pre-training, offline adaptation, and reinforcement learning. In this repo, all the stages are provided, together with the pre-trained checkpoint and the adapted checkpoints.

We also encourage the user to use our pre-trained representations directly for their own downstream tasks. 


To benchmark our method, we also provide several strong baselines in this repo, including [VC-1](https://github.com/facebookresearch/eai-vc), [MVP](https://github.com/ir413/mvp), [R3M](https://github.com/facebookresearch/r3m), and [RRL](https://github.com/facebookresearch/RRL).


*Enjoy Dexterity!*


# 💻 Installation
See [INSTALL.md](INSTALL.md).

We also provide some error catching solutions in [INSTALL.md](INSTALL.md). 

Feel free to post an issue if you have any questions.



# 🛠️ Usage 
We use `wandb` to log the training process. Remember to set your `wandb` account before training by `wandb login`. You could also disable `wandb` by `use_wandb=0` in our script.


Given a task name `task_name`, you could run the following pipeline.
- **Stage 1: Human Hand Pretraining.**
  - Simply download the pre-trained hand representation from [FrankMocap](https://github.com/facebookresearch/frankmocap) by this command
    ```bash
    wget https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/checkpoints_best/pose_shape_best.pth -O archive/frankmocap_hand.pth --no-check-certificate`
    ```
- **Stage 2: Offline Adaptation.**
  - First, download the initial model weights in Stage 2 from [here](https://drive.google.com/file/d/1vFB4_u21fA9qjQFReKvckTCyFL4K_yAi/view?usp=sharing) and put it under `stage2_adapt/`.
  - Second, generate image dataset for offline adaptation. See `scripts/adroit/gen_img_dataset.sh` or `scripts/dexmv/gen_img_dataset.sh` for details. An example:
    ```bash
    bash scripts/adroit/gen_img_dataset.sh hammer
    ```
  - Third, adapt affine transformation in pretrained model. See `scripts/train_stage2.sh` for details. An example:
    ```bash
    bash scripts/train_stage2.sh hammer-v0
    ```
  - For the users' convenience, we also provide the adapted checkpoints for all the tasks. You can download them from [here](https://drive.google.com/file/d/1TPCAnZMOcgErMruZ45R8MAg7b4iWu5DT/view?usp=sharing) and put them under `archive/` folder.
- **Stage 3: Reinforcement Learning.**
  - Train RL agents with the pre-trained representations. See `scripts/adroit/train.sh` or `scripts/dexmv/train.sh` for details. An example:
    ```bash
    bash scripts/adroit/train.sh hammer hindex test 0 0
    ```
    Arguments are task name, representation name, experiment name, seed, and GPU id respectively.  


# 🦉 Tasks
We provide 12 dexterous manipulation Tasks in total:
- Adroit (3): pen, door, hammer
- DexMV (9): pour, place_inside, relocate-mug, relocate-foam_brick, relocate-large_clamp, relocate-mustard_bottle, relocate-potted_meat_can, relocate-sugar_box, relocate-tomato_soup_can


# 🙏 Acknowledgement
Our work is based on many open-source projects. The algorithms are mainly built upon [RRL](https://github.com/facebookresearch/RRL) and [TTP](https://github.com/harry11162/TTP). The simulation environments are from [DAPG](https://github.com/aravindr93/hand_dapg) and [DexMV](https://github.com/yzqin/dexmv-sim). The pre-trained hand representation is from [FrankMocap](https://github.com/facebookresearch/frankmocap). Baselines are from [RRL](https://github.com/facebookresearch/RRL), [MVP](https://github.com/ir413/mvp), [R3M](https://github.com/facebookresearch/r3m) and [VC-1](https://github.com/facebookresearch/eai-vc). We thank all these authors for their nicely open sourced code and their great contributions to the community.


# 🏷️ License
H-InDex is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.


# 📝 Citation
If you find our work useful, please consider citing:
```
@article{Ze2023HInDex,
  title={H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation},
  author={Yanjie Ze and Yuyao Liu and Ruizhe Shi and Jiaxin Qin and Zhecheng Yuan and Jiashun Wang and Xiaolong Wang and Huazhe Xu},
  journal={NeurIPS}, 
  year={2023},
}
```


