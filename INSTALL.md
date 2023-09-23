# 1 Create a New Conda Environment
Simply run the following commands to create a new conda environment.
```bash
conda create -n dex python=3.8
conda activate dex
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install opencv-python matplotlib scikit-image pandas natsort ipdb yacs wandb gpustat hydra-core==1.1.0
pip install Cython==0.29.35
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
cd stage3_RL
pip install -e. 
cd mjrl
pip install -e. 
cd ..
cd mj_envs
pip install -e.
cd ../..
cd third_party/dexmv-sim/
pip install -e .
cd ../..
```
In order to let other researchers better study this direction, we not only open source our algorithm, but also all the baselines we use.
```bash
# R3M
cd third_party/r3m
pip install -e .
cd ..
# MVP
cd mvp
pip install -e .
cd ..
# VC-1
cd eai-vc/vc_models
pip install -e .
cd ../../..
```

To use these baseline methods, you may need to also download their checkpoints:
- [VC-1](https://github.com/facebookresearch/eai-vc) (baseline): download [checkpoint](https://dl.fbaipublicfiles.com/eai-vc/vc1_vitb.pth) and put it under `archive/`.
- [R3M](https://github.com/facebookresearch/r3m) (baseline): download [model url](https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA) and [config url](https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8), and put them under `archive/r3m_ckpt/`.
- [MVP](https://github.com/ir413/mvp) (baseline): download [checkpoint](https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth) and put it under `archive/`


Finally, you can try to run visual RL experiments using our scripts. If you have any questions, please feel free to post an issue in Github or contact us via email.

# Error Catching
1. pip no response: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
2. fatal error: GL/osmesa.h: No such file or directory: `sudo apt-get install libosmesa6-dev`
3. FileNotFoundError: [Error 2] No such file or directory: 'patchelf': `sudo apt-get install patchelf`
4. ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory: 
    - first solution: `sudo apt-get install libpython3.7`
    - second solution: `export LD_LIBRARY_PATH=/home/yanjieze/miniconda3/envs/mvp/lib`
5. error when compiling mujoco:
```
Error compiling Cython file:
------------------------------------------------------------
...
    See c_warning_callback, which is the C wrapper to the user defined function
    '''
    global py_error_callback
    global mju_user_error
    py_error_callback = err_callback
    mju_user_error = c_error_callback
                     ^
------------------------------------------------------------
```
solution:
```bash
pip install Cython==0.29.35
```