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

## Documents
- [Run DiffSinger (SVS version)](docs/README-SVS.md).


 



## Citation
    @article{liu2021diffsinger,
      title={Diffsinger: Singing voice synthesis via shallow diffusion mechanism},
      author={Liu, Jinglin and Li, Chengxi and Ren, Yi and Chen, Feiyang and Liu, Peng and Zhao, Zhou},
      journal={arXiv preprint arXiv:2105.02446},
      volume={2},
      year={2021}}


    