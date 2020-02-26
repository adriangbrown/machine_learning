import torch, torchvision
import argparse
import json
import random
import os
#import seaborn as sns
import numpy as np
from torch import nn
import datetime
from torch import optim
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F
#from matplotlib import pyplot as plt
from torchvision import datasets, transforms, models


def arg_parser():
    parser = argparse.ArgumentParser(description='Allow user choices')
    # Add optional architecture argument
    parser.add_argument('--load_dir', type=str, help='Save to location for re-use')
    parser.add_argument('--map', action='store_true', help='Augment mappings')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
    parser.add_argument('--image', type=str, help='Sample Image')

    args = parser.parse_args()
    return args

def cat_mapper(map):
    if type(map) == type(None):
        print('default mapping')
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(model_path):
    checkpoint = torch.load(model_path)
    model = models.vgg16(pretrained=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']

    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    # Resize
    im = im.resize((256,256))
    # Crop finding delta of 256 versus 224
    amount_in = (256-224)/2
    # Use indent for cropping
    im = im.crop((amount_in, amount_in, 256-amount_in, 256-amount_in))
    # Set array and convert to float
    im = np.array(im)/255
    means = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    im = (im-means)/std_dev
    im = im.transpose(2,0,1)
    return(im)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax
# TODO: Implement the code to predict the class from an image file

def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
       
    model.eval();
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")
    log_probs = model.forward(torch_image)
    linear_probs = torch.exp(log_probs)
    top_probs, top_labels = linear_probs.topk(5)
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    return top_probs, top_labels, top_flowers
    
def main():
    args = arg_parser()

    model = load_checkpoint(args.load_dir)

    img_path = "flowers/test/19/image_06175.jpg"

    cat_to_name = cat_mapper(args.map)
    flower_num = img_path.split('/')[2]
    flower_title = cat_to_name[flower_num]
    
    img = process_image(img_path)
    
    probs, labs, flowers = predict(img_path, model) 
    print(probs, flowers)
    
if __name__ == '__main__':
    main()
