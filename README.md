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
 

