# DMLCC

- Code for the paper: [Deep Multi-Level Contrastive Clustering for Multi-Modal Remote Sensing Images | Proceedings of the 33rd ACM International Conference on Multimedia](https://dl.acm.org/doi/10.1145/3746027.3755073)

  

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{10.1145/3746027.3755073,
author = {Liu, Weiqi and Zhang, Yongshan and Wang, Xinxin and Zhang, Lefei},
title = {Deep Multi-Level Contrastive Clustering for Multi-Modal Remote Sensing Images},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3755073},
doi = {10.1145/3746027.3755073},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {1239–1247},
numpages = {9},
keywords = {contrastive learning, cross-modal fusion, deep multi-modal clustering, multi-modal remote sensing},
location = {Dublin, Ireland},
series = {MM '25}
}
```



## System and Version

- Linux system
- Python 3.8 or higher
- PyTorch 2.0 or higher 
- CUDA 11.8 recommended

------

## Environment Setup

In the following steps, PyTorch **with CUDA 11.8** is used by default. If your version is different, please modify accordingly.

### Detailed Instructions

Install PyTorch with CUDA 11.8, and create a CUDA toolkit that is only available to the current virtual environment.

Install Mamba and related dependencies:

```
pip install causal-conv1d==1.2.0.post2
pip install mamba-ssm==1.2.0.post1
```

Install other model dependencies. （Some dependencies come bundled with PyTorch. If any are missing, please install them according to the error messages.）

```
pip install scipy
pip install scikit-learn
pip install spectral
pip install timm
pip install munkres
```



## Get Started

```
run main.py
```



# Acknowledgements

- This code is partially inspired by the [TMPCC](https://github.com/AngryCai/TMPCC) repository.

