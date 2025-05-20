<h1 align="center">
MAARDTI: A multi-perspective attention aggregating model for predicting drug-target interaction
</h1>

![workflow](https://github.com/user-attachments/assets/2b4de85e-eb20-4d2f-8302-6f918443f26a)


## Requirments
* pytorch >=1.2
* numpy
* sklearn
* tqdm

## Environment
Try the following command for installation. 
```sh
# Install Python Environment
conda env create -f environment.yml
conda activate MAARDTI
```

## Installation
- You can install the required libraries `environment.yml`
- If you encounter any installation errors, please don't hesitate to reach out to us for assistance.

## Download datasets
Datasets (DrugBank, Davis, KIBA) are provided by project [MCANet](https://github.com/MrZQAQ/MCANet/tree/main).
Cold Drug/Target/Binding are provided by project [DLM-DTI](https://github.com/jonghyunlee1993/DLM-DTI_hint-based-learning/tree/master). 

## Sample test
We provided sample scripts for easily training by MAARDTI.
```python default
python start.py data=Sample c_p=256 c_d=8 outpath='samle_p256_d8'
```

## Training and Testing
We provided scripts for easily training by MAARDTI.
```python default
python start.py ds=Davis outpath='sample' epoch=300 c_p=16 c_d=8'
```

