# Image Classifier with Pytorch

## Description:  Training and Validation application that is used to predict types of images

Steps include: 1) loaded and pre-processed 2) model is trained and validated 3) model is then used 
to predict a type of flower

Files
*ipython notebook:  Image Classifier Deep Learning Notebook
*train.py - Used to load datam, train model, and save model checkpoint
*predict.py - Takes saved model and applies to new image data, predicting image type

## Use train.py
```bash
python train.py --arch <Pick architecture> \
                --data_dir <Folder location of images> \
                --save_dir'<checkpoint folder> \
                --gpu <Use GPU for processing \
                --epochs <Define epoch intervals> \ 
                --learning_rate <Model Learning Rate> \
                --hidden_units <Number of hidden units> 
```
                
## train.py Example
```bash
python train.py --arch vgg11 --epochs 3
```

## Use predict.py
```bash
python predict.py --load_dir <Load from location for re-use> \
                  --map <Label Mappings> \
                  --topk <Number of top choices to display> \
                  --gpu <Use GPU for processing> \
                  --image <Sample Image location> 
```

## predict.py Example
```bash
python predict.py --topk 5 --load_dir checkpoint.pth
```

## Python Libraries
- torch
- torchvision
- argparse
- random
- os
- os.path
- json
- seaborn
- numpy 
- datetime
- collections
- PIL 
- matplotlib

## Adrian Brown
adrian.g.brown@gmail.com
