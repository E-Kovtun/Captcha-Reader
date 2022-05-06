# Captcha-Reader

The goal of this task is to train deep learning model that takes as input captcha image and outputs symbols which are depicted in this capctha. 
The example of captcha image:


![Capctha_Example](/images/captcha_example.png)

# 1. Download initial dataset
The following command will download the dataset in the folder `../dataset/`:

``` bash scripts/download_init_dataset.sh```

# 2. Splitting dataset
To split the dataset in train, validation, and test sets run the command:

```python3 data_splitter.py```

