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

    parser.add_argument('--load_dir', type=str, help='Load from location for re-use')
    parser.add_argument('--map', type=str, default = 'cat_to_name.json', help ='Augment mappings')
    parser.add_argument('--topk', type=int, default=2, help='Top choices')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU for processing')
    parser.add_argument('--image', type=str, default = 'flowers/test/19/image_06155.jpg', help='Sample Image')

    args = parser.parse_args()
    return args

def cat_mapper(filename):
    with open(filename, 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(model_path):
    '''
    Args:
      model_path(str) default or user input checkpoint
    Returns:
      model (dict)
    
    '''
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    
    '''
    # Resize
    im = image.resize((256,256))
    # Crop finding delta of 256 versus 224
    amount_in = (256-224)/2
    # Use indent for cropping
    im = im.crop((amount_in, amount_in, 256-amount_in, 256-amount_in))
    # Set array and convert to float
    im = np.array(im)/255
    means = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    im = (im-means)/std_dev
    norm_image = im.transpose(2,0,1)
    return norm_image

def predict(model, np_image, topk, gpu):
    """Predict the class(es) of image using trained deep learning model
    
    Args:
        model(torchvision.models)
        img_tensor (torch)
        in_args_topk (int): User input
        in_args_gpu (bool): User input
 
    """

    # Convert image to tensor
    np_image = np.expand_dims(np_image, axis=0)
    img_tensor = torch.from_numpy(np_image)
    # Convert to float
    float_tensor = img_tensor.type(torch.FloatTensor)
    
    image_pred = float_tensor.to('cuda')
   
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_pred)
  
    probs = torch.exp(outputs) 
    # Find topk 
    probs, labels = probs.topk(k=topk) 
    # From torch to numpy to lists
    if gpu == True:
        probs, labels = probs.to('cpu'), labels.to('cpu')
    probs, labels = probs.numpy(), labels.numpy()
    probs, labels = probs[0].tolist(), labels[0].tolist()
    
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    # Map labels to class in index\
    labels = [idx_to_class[lab] for lab in labels]
    return probs, labels
    
def main():
    args = arg_parser()

    model = load_checkpoint(args.load_dir)

    img_path = args.image
    
    image = Image.open(img_path)
        
    np_image = process_image(image)
    cat_to_name = cat_mapper(filename=args.map)

    probs, labels = predict(model, np_image, args.topk, args.gpu) 
    print(probs, labels)
    
    labels = [cat_to_name[key] for key in labels]
    
    print('\n Filepath to image: ', img_path, '\n',
          '\n  Top labels: ', labels,
          '\n  Probabilities: ', probs)
    
if __name__ == '__main__':
    main()
