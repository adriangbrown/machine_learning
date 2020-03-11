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

def image_to_tensor(np_image):
    np_image = np.resize(np_image,(1, 3, 224, 224))
    
    # NumPy to torch
    img_tensor = torch.from_numpy(np_image)
    img_tensor = img_tensor.type(torch.FloatTensor)
    
    return img_tensor

def predict(model, image_tensor, topk, gpu):
    """Predict the class(es) of image using trained deep learning model
    
    Args:
        model(torchvision.models)
        img_tensor (torch)
        in_args_topk (int): User input
        in_args_gpu (bool): User input
 
    """
    if gpu == True:
        model.to('cuda')
        image_tensor = image_tensor.to('cuda')
    # Run model in evaluation mode
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    # From softmax to probabilities
    probs = torch.exp(outputs.data) 
    # Find topk probabilities and indices 
    top_probs, indices = probs.topk(k=topk) 
    # From torch to numpy to lists
    if gpu == True:
        top_probs, indices = top_probs.to('cpu'), indices.to('cpu')
    top_probs, indices = top_probs.numpy(), indices.numpy()
    top_probs, indices = top_probs[0].tolist(), indices[0].tolist()
    # Find the class using the indices (reverse dictionary first)
    idx_to_class =  {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in indices]
    top_flowers = [idx_to_class[lab] for lab in indices]
    
    return top_probs, top_labels, top_flowers
    
def main():
    args = arg_parser()

    model = load_checkpoint(args.load_dir)

    img_path = args.image
    
    image = Image.open(img_path)
        
    np_image = process_image(image)
    
    image_tensor = image_to_tensor(np_image)

    probs, top_labels, top_flowers = predict(model, image_tensor, args.topk, args.gpu) 
    print(probs, top_labels, top_flowers)
    
    class_name_dict = cat_mapper(filename=args.map)
    flower_names = [class_name_dict[key] for key in top_flowers]
    labels = [class_name_dict[key] for key in top_labels]
    
    print('\n Filepath to image: ', img_path, '\n',
          '\n  Top labels: ', top_labels,
          '\n  Flower names: ', flower_names,
          '\n  Probabilities: ', probs)
    
if __name__ == '__main__':
    main()
