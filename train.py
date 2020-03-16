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

def arg_parser():
    '''Function creates series of arguments for use in modules
    
    '''
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
        train, valid and test data locations
    """
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return train_dir, valid_dir, test_dir
    
    
def data_transformer():
    """Return dictionary of data transforms for the 
    training, validation, and testing sets
    Args:
        None
    Returns:
        training, validation, and testing transformations
        
    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])
    
    return train_transforms, valid_transforms, test_transforms
    
def data_loader(train_dir, valid_dir, test_dir, train_transforms, valid_transforms, test_transforms):
    """Load datasets and define the dataloaders
    
    Args:
        train_dir, valid_dir, test_dir:  location of data
        train_transforms, valid_transforms_test_transforms:  data transformation specs
    Returns:
         train_loader, valid_loader, test_loader:  data loading specifications
         class_to_idx_dict (dict): Mapping from class number to tensor 
            index
    """
    # ImageFolder data loading
    train_data = datasets.ImageFolder(
        train_dir, 
        transform=train_transforms
        )
    valid_data = datasets.ImageFolder(
        valid_dir, 
        transform=valid_transforms
        )
    test_data = datasets.ImageFolder(
        test_dir, 
        transform=test_transforms
        )
   
    # Flower folder number to index mapping
    class_to_idx_dict = train_data.class_to_idx
    
    # Perform data loading
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=32, shuffle=True
        )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, 
        batch_size=32, shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=32, shuffle=True
        )
    
    return train_loader, valid_loader, test_loader, class_to_idx_dict

def arch_choice(arch):
    '''Returned pre-trained pytorch model
    
    Args: 
      arch (str): user input
    Return: 
      model(torchvision.models): Pretrained network
    
    '''
    model = getattr(models, arch)
    return model(pretrained=True)
             
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
        ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.55)),
        ('fc2', nn.Linear(hidden_units, 102, bias=True)),
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

    valid_loss = 0
    print_every = 50
    correct = 0
    population = 0
    accuracy = 0
    
    with torch.no_grad():
        for data in validloader:
            inputs, labels = data
            if gpu == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            output = model(inputs)
            valid_loss += criterion(output, labels).item()
        
            ps = torch.exp(output)
            match = (labels.data == ps.max(dim=1)[1])
            accuracy += match.type(torch.FloatTensor).mean()
            '''
            _, predicted = torch.max(output.data, 1)
            valid_total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if steps % print_every == 0:
                print('Validation Loss: {:.4f}'.format(valid_loss/print_every))
                running_loss = 0
            '''
    print('Accuracy', accuracy/len(validloader))
                
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
    
    train_dir, valid_dir, test_dir = data_folders(args.data_dir)
    train_transforms, valid_transforms, test_transforms = data_transformer()
    
    train_loader, valid_loader, test_loader, class_to_idx_dict = data_loader(train_dir, valid_dir, test_dir, train_transforms, valid_transforms, test_transforms)
    
    # Import canned pre-trained model
    model = arch_choice(args.arch)
    model.class_to_idx = class_to_idx_dict

    #Freeze parameters
    for param in model.parameters():
      param.requires_grad = False

    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)

    trainer(model, train_loader, args.epochs, args.learning_rate, args.gpu)
    validation(model, valid_loader, args.gpu)
    
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