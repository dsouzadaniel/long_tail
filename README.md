# A Tale Of Two Long Tails
Official :octocat: Codebase  for [A Tale Of Two Long Tails](https://arxiv.org/abs/2107.13098)

**If you use this software, please consider citing:**

```@article{d2021tale,
  title={A Tale Of Two Long Tails},
  author={D'souza, Daniel and Nussbaum, Zach and Agarwal, Chirag and Hooker, Sara},
  journal={arXiv preprint arXiv:2107.13098},
  year={2021}}
```

![Long Tail Examples](https://i.ibb.co/ngLndM8/longtail-example.png) :rotating_light: **_tldr: Examples of Atypical and Noisy Error. The former is reducible with the introduction of information and the other is not!_** :rotating_light:


# **Setup**
This repository is built using PyTorch:fire:. You can install the necessary libraries by 
`pip install -r requirements.txt`

# **Datasets**
1. Download [CIFAR-10/CIFAR-100 LongTail Datasets](https://drive.google.com/drive/folders/1DXTHDuF81OfZ2wIjsrF13wlqnORa9IPo?usp=sharing)
2. Unzip above files in folder "datasets" in main directory

# **Usage**
The scripts to train CIFAR-10/CIFAR-100 models on all datasets is train_c10.py/train_c100.py.

**Training**

1. Set Variable _**MSP_AUG_PCT**_ to a value between (0,1). This controls how much of the dataset to augment based on the MSP.Default is  0.2 ( _Targeted Augment Variant_ )

2. Set Variable _**TRAIN_DATASET**_ to either 'cifar10'(Original), 'N20_A20_T60'(C-Score), 'N20_A20_TX2'(Frequency)

3. Run `python train_c10.py` to train CIFAR-10 models

The above steps can be repeated for CIFAR-100 by using train_c100.py

# **Results**

__CIFAR-10__
![alt text](https://i.ibb.co/jhwPndc/c10.png)

__CIFAR-100__
![alt text](https://i.ibb.co/Rpw4fK7/c100.png)

_Visualization Code will be added shortly._

# **Licenses**
Note that the code in this repository is licensed under MIT License. Please carefully check them before use.
# **Questions?**
If you have questions/suggestions, please feel free to [email](mailto:ddsouza@umich.edu) or create github issues.

