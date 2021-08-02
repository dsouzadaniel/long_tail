# long_tail
Official Codebase for "A Tale Of Two Long Tails"(https://arxiv.org/abs/2107.13098)

**Setup** 

1. Download CIFAR-10/CIFAR-100 LongTail Datasets here : https://drive.google.com/drive/folders/1DXTHDuF81OfZ2wIjsrF13wlqnORa9IPo?usp=sharing
2. Unzip above files in folder "datasets" in main directory
3. Install requirements
    `pip install -r requirements.txt`
    
**Training**

1. Set Variable _**MSP_AUG_PCT**_ to a value between (0,1). This controls how much of the dataset to augment based on the MSP.Default is  0.2 ( _Targeted Augment Variant_ )

2. Set Variable _**TRAIN_DATASET**_ to either 'cifar10'(Original), 'N20_A20_T60'(C-Score), 'N20_A20_TX2'(Frequency)

3. Run `python train_c10.py` to train CIFAR-10 models

Visualization Code will be added shortly. Reach out to **ddsouza[at]umich[dot]edu** with any questions
