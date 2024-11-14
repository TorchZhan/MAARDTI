<h1 align="center">
MAAR: A multi-perspective attention aggregating model for predicting drug-target interaction
</h1>

![workflow](https://github.com/user-attachments/assets/15ae4eaf-9774-48c6-a6e9-f4c57da1b52e)

## Requirments
* pytorch >=1.2
* numpy
* sklearn
* tqdm
* 

## Environment
Try the following command for installation. 
```sh
# Install Python Environment
conda env create -f environment.yml
conda activate MAARDTI
```

## Installation
- You can install the required libraries by running `pip install -r requirements.txt`
- If you encounter any installation errors, please don't hesitate to reach out to us for assistance.

## Download datasets
Datasets (DrugBank, Davis, KIBA) are provided by project [MCANet](https://github.com/MrZQAQ/MCANet/tree/main).
Cold Drug/Target/Binding are provided by project [DLM-DTI](https://github.com/jonghyunlee1993/DLM-DTI_hint-based-learning/tree/master). 

## Training and Testing
We provided scripts for easily training by MAARDTI.
```python default
python train.py ds=Davis outpath='sample' epoch=300 c_p=16 c_d=8'
```

