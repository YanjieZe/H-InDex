# 1 Create a New Conda Environment
Simply run the following commands to create a new conda environment.
```bash
conda create -n dex python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install opencv-python matplotlib scikit-image pandas natsort ipdb yacs wandb gpustat hydra-core==1.1.0
```

# 2 Install MuJoCo
To install mujoco, you need to first download the file from this [link](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) and `tar -xvf the_file_name` in the `~/.mujoco` folder. Then, install the python packages:
```bash
pip install mujoco-py
```
After that, add the following lines to your `~/.bashrc` file:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/YOUR_PATH_TO_THIS/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
Remember to `source ~/.bashrc` to make the changes take effect.

# 3 Install Some Modified Packages and Baselines
We modified some packages to make them compatible with our code. You can install them by running the following commands:
```bash
cd RRL
pip install -e. 
cd mjrl
pip install -e. 
cd ..
cd mj_envs
pip install -e.
cd ../..
cd dexmv-sim
pip install -e .
cd ..
```
In order to let other researchers better study this direction, we not only open source our algorithm, but also all the baselines we use.
```bash
# R3M
cd r3m
pip install -e .
cd ..
# MVP
cd mvp
pip install -e .
cd ..
# VC-1
cd eai-vc/vc_models
pip install -e .
cd ../..
```
To use Stage 2 adaptation or baseline methods, you may need to also download their checkpoints:
- [VC-1](https://github.com/facebookresearch/eai-vc) (baseline): download [checkpoint](https://dl.fbaipublicfiles.com/eai-vc/vc1_vitb.pth) and put it under `archive/`.
- [R3M](https://github.com/facebookresearch/r3m) (baseline): download [model url](https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA) and [config url](https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8), and put them under `archive/r3m_ckpt/`.
- [MVP](https://github.com/ir413/mvp) (baseline): download [checkpoint](https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth) and put it under `archive/`
- Initial model weights in Stage 2  (human pose estimation model): download [checkpoint](https://drive.google.com/file/d/1vFB4_u21fA9qjQFReKvckTCyFL4K_yAi/view?usp=sharing) and put it under `stage2_adapt/`


Finally, you can try to run visual RL experiments using `scripts/dexmv/train.sh`. If you have any questions, please feel free to post an issue in Github or contact us via email.

# Error Catching
1. pip no response: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
2. fatal error: GL/osmesa.h: No such file or directory: `sudo apt-get install libosmesa6-dev`
3. FileNotFoundError: [Error 2] No such file or directory: 'patchelf': `sudo apt-get install patchelf`
4. ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory: 
    - first solution: `sudo apt-get install libpython3.7`
    - second solution: `export LD_LIBRARY_PATH=/home/yanjieze/miniconda3/envs/mvp/lib`