# Captcha-Reader

The goal of this task is to train deep learning model that takes as input captcha image and outputs symbols which are depicted in this capctha. 
Example of captcha image:

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
 
 Alternatively, you can download already generated images with the command:
 
 ```bash scripts/download_synthetic_datasets.sh```
 
 Example of synthetic captcha image:
 
 ![Synthetic Capctha_Example](/images/synthetic_captcha_example.png)
 
 # 4. Train model
 First, we pretrain a model on generated synthetic dataset. As a feature extractor in our model we use ResNet-18. All parametrs for model pretraining are defined in `configs/pretraining.json`. To launch pretraining process:
 
 ```python3 train.py pretrain```
 
 The checkpoint will be saved in `./results/` folder.
 
 Second, we need to finetune our pretrained model on the given dataset. For this, run:
 
 ```python3 train.py finetune```
 
All parametrs for model finetuning can be changed in `configs/finetuning.json`. The checkpoint will be saved in `./results/` folder.

In order to download the model checkpoints and not train the model from scratch, run: 

```bash scripts/download_checkpoints```

As a result, in a folder `./results/` you will have `resnet18_synthetic_checkpoint.pt` - checkpoint for pretrained model and `resnet18_finetuned_checkpoint.pt` - final checkpoint of finetuned model. 
 
 # 5. Evaluate model
 
 
 
 
 
 
 
 
 
 
 

