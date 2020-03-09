import torch, torchvision
import argparse
import random
import os
import os.path
import json
import seaborn as sns
import numpy as np
from torch import nn
import datetime
from torch import optim
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import datasets, transforms, models

# USAGE
# python train.py /flowers
#  --arch <network architecture>
#  --save_dir <checkpoint folder>
#  --gpu
#  --epoch <epoch count>
#  --hidden_units <hidden units pre-classifier>


def arg_parser():
    parser = argparse.ArgumentParser(description='Allow user choices')
    # Add optional architecture argument
    parser.add_argument('--arch', type=str, default='vgg16', help='Pick architecture')
    parser.add_argument('--data_dir', type=str, default='flowers/', help='Folder location of images')
    parser.add_argument('--save_dir', type=str, default=None, help='checkpoint folder')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU for processing')
    parser.add_argument('--epochs', type=int, default=3, help='Define epoch intervals')
    parser.add_argument('--learning_rate', type=float, default=.003, help='Learning Rate Entry')

    parser.add_argument('--hidden_units', type=int, default=512, help='Define number of hidden units')

    args = parser.parse_args()
    print(args) 
    return args

def data_folders(data_dir):
    """Returns dictionary of image folders for train, validation and test sets
    Args:
        data_dir (str): path to images
    Returns:
        data_folder_dict: paths to data folders
    """
    
    data_folder_dict = dict()   
    data_folder_dict['train'] = data_dir + '/train'
    data_folder_dict['valid'] = data_dir + '/valid'
    data_folder_dict['test'] = data_dir + '/test'
    return data_folder_dict
    
    
def data_transformer():
    """Return dictionary of data transforms for the 
    training, validation, and testing sets
    Args:
        None
    Returns:
        transforms_dict (dict): Pipelines of data transforms for the 
            training, validation, and testing sets
    """
    transforms_dict = dict()
    transforms_dict['train'] = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])
    transforms_dict['valid'] = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])
    transforms_dict['test'] = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])
    
    return transforms_dict
    
def data_loader(data_folder_dict, transforms_dict):
    """Load datasets and define the dataloaders
    
    Args:
        data_folder_dict (dict):  paths to data sets 
        transforms_dict (dict): data transformations
    Returns:
         dataloader_dict (dict): Dataloaders
         class_to_idx_dict (dict): Mapping from class number to tensor 
            index
    """
    # Load the datasets with ImageFolder
    datasets_dict = dict()
    
    datasets_dict['train'] = datasets.ImageFolder(
        data_folder_dict['train'], 
        transform=transforms_dict['train']
        )
    datasets_dict['valid'] = datasets.ImageFolder(
        data_folder_dict['valid'], 
        transform=transforms_dict['valid']
        )
    datasets_dict['test'] = datasets.ImageFolder(
        data_folder_dict['test'], 
        transform=transforms_dict['test']
        )
   
    # Flower folder number to index mapping
    class_to_idx_dict = datasets_dict['train'].class_to_idx
    
    # Using the image datasets and the trainforms, define dataloaders
    dataloaders_dict = dict()
    dataloaders_dict['train'] = torch.utils.data.DataLoader(
        datasets_dict['train'], 
        batch_size=32, shuffle=True
        )
    dataloaders_dict['valid'] = torch.utils.data.DataLoader(
        datasets_dict['train'], 
        batch_size=32, shuffle=True
        )
    dataloaders_dict['test'] = torch.utils.data.DataLoader(
        datasets_dict['train'], 
        batch_size=32, shuffle=True
        )
    
    return dataloaders_dict, class_to_idx_dict

def arch_choice(arch):
    '''Returned pre-trained pytorch model
    
    Args: 
      arch (str): user input
    Return: 
      model(torchvision.models): Pretrained network
    
    '''
    model = getattr(models, arch)
    return model(pretrained=True)
      
def cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        print(cat_to_name)
    return cat_to_name

def model_forward(arch='vgg16'):
    if type(arch) == type(None):
        model = models.vgg16(pretrained=True)
        print('default model = vgg16')
    else:
        exec('model = models.{}(pretrained=True)'.format(arch))
 
    return model
             
def initial_classifier(model, hidden_units):
    '''Return classifier layer
    
    Args:
      model:  pytorch model - pretrained
      hidden units (int): number of hidden layer units
    Returns:
      classifer
      
    '''
    input_features = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([
        ('dropout_1', nn.Dropout(0.55)),
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.55)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
         ]))
    
    return classifier

def validation(model, validloader, gpu):
    '''
    Args:
      model:  pytorch pre-trained model
      validloader:  data loader
      device: user input cpu or gpu
    
    '''
    criterion = nn.NLLLoss()

    running_loss = 0
    print_every = 50
    steps = 0
    valid_correct = 0
    valid_total = 0
    
    with torch.no_grad():
        for data in validloader:
            steps += 1
            images, labels = data
            if gpu == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            output = model(images)
            running_loss += criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
            if steps % print_every == 0:
                print('Validation Loss: {:.4f}'.format(running_loss/print_every))
                running_loss = 0
    print('Validation Accuracy: {1:.1%} \n {0:d} validation Images'.format(valid_total, valid_correct / valid_total))
                
def trainer(model, trainloader, epochs, learning_rate, gpu):
    '''Train the model
    
    Args:
      model: pre-trained model
      trainloader:  dataloader
      epochs (int): user input epochs
      learning_rate (float): user input learning_rate
      gpu (bool): user input
      
    '''
    if gpu == True:
        model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), 
                            lr=learning_rate)
    print_every = 50
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if gpu == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print('Epoch: {0}/{1}'.format(e+1,epochs),
                  'Train loss: {:.2f}'.format(running_loss/print_every))
                running_loss = 0

def main():
    args = arg_parser()
    
    data_folder_dict = data_folders(args.data_dir)
    transforms_dict = data_transformer()
    
    dataloaders_dict, class_to_idx_dict = data_loader(data_folder_dict, transforms_dict)
    
    # Import canned pre-trained model
    model = arch_choice(args.arch)
    model.class_to_idx = class_to_idx_dict

    #Freeze parameters
    for param in model.parameters():
      param.requires_grad = False

    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)

    trainer(model, dataloaders_dict['train'], args.epochs, args.learning_rate, args.gpu)
    validation(model, dataloaders_dict['valid'], args.gpu)
    
    checkpoint = {
        'model': model,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
     
    if args.save_dir is None:
        torch.save(checkpoint, 'checkpoint.pth')
    else:
        torch.save(checkpoint, args.save_dir + 'checkpoint.pth')
          
    
if __name__ == '__main__':
    main()