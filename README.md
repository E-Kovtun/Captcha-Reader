# Captcha-Reader

The goal of this task is to train deep learning model that takes as input captcha image and outputs symbols which are depicted in this capctha. 
The example of captcha image:


![Capctha_Example](/images/captcha_example.png)

# 1. Download initial dataset
The following command will download the dataset in the folder `../dataset/`:

``` bash scripts/download_init_dataset.sh```

# 2. Splitting dataset
To split the dataset into train, validation, and test sets run the command:

```python3 data_splitter.py```

This python script will create the folders `../prepared_datasets/train_dataset/`, `../prepared_datasets/valid_dataset/`, and `../prepared_datasets/test_dataset/`. 

* Train size: 550 images
* Valid size: 120 images
* Test size: 400 images

Alternatively, you can run the script that downloads already prepared datasets:

 ```bash scripts/download_prepared_datasets.sh```
 
 # 3. Generating synthetic captcha dataset
 There are too few captcha images in the inital dataset to train an effictive model. Thus, we will generate synthetic captcha images for the model pretraining. For this, run:
 
 ```python3 generate_synthetic_captcha.py```
 
 This python script will create the folders: `../synthetic_datasets/train_synthetic/` and `../synthetic_datasets/valid_synthetic/` with 100.000 and 10.000 synthetic images correspondingly. 
 
 Alternatively, you can download alreade negerated images with the command:
 
 ```bash scripts/download_synthetic_datasets.sh```
 
 
 
 
 
 
 
 
 
 
 
 

