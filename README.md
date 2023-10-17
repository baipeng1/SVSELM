# SVSELM
This is the implementation code of the paper "Improving Chinese Pop Song and Hokkien Gezi Opera Singing Voice Synthesis by Enhancing Local Modeling".

We are particularly grateful to the [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger) project as we have made modifications and conducted experiments based on their code.

We have built a Gezi Opera dataset. If you are only using it for research, you can send an email to me and I will apply for authorization from teacher. We have an agreement with the data recording people.

[Demo page](https://htmlpreview.github.io/?https://github.com/baipeng1/SVSELM/blob/main/demo/index.html)


We have modified the original file to make it more concise.

## Environments
1. If you want to use env of anaconda:
    ```sh
    conda create -n your_env_name python=3.8
    source activate your_env_name 
    pip install -r requirements_2080.txt   (GPU 2080Ti, CUDA 10.2)
    or pip install -r requirements_3090.txt   (GPU 3090, CUDA 11.4)
    ```

2. Or, if you want to use virtual env of python:
    ```sh
    ## Install Python 3.8 first. 
    python -m venv venv
    source venv/bin/activate
    # install requirements.
    pip install -U pip
    pip install Cython numpy==1.19.1
    pip install torch==1.9.0
    pip install -r requirements.txt
    ```

## Running 
Taking Gezi Opera as an example:

### 1. Preparation
#### Data Preparation
a) Download and extract Gezi Opera dataset, then create a link to the dataset folder: `ln -s /xxx/gezixi/ data/processed/gezixi`

b) Run the following scripts to pack the dataset for training/inference.
```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/gezixi.yaml
# `data/binary/popcs-pmf0` will be generated.
```

#### Vocoder Preparation

We use pre-trained model of [HifiGAN-Singing](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip), and this model is trained by DiffSinger team.
Please unzip this file into `checkpoints` before training your acoustic model.

### 2. Training Example


```sh
# First, train SVSELM;
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/gezixi.yaml --exp_name gezixi --reset
# Then, infer SVSELM;
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/gezixi.yaml --exp_name gezixi --reset --infer 
```

