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
    parser = argparse.ArgumentParser(description='Allow user choices')
    # Add optional architecture argument
    parser.add_argument('--arch', type=str, help='Pick architecture')
    parser.add_argument('--save_dir', type=str, help='Save to location for re-use')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
    parser.add_argument('--epochs', type=int, help='Define epoch intervals')
    parser.add_argument('--hidden_units', type=int, help='Define number of hidden units')

    args = parser.parse_args()
    return args
    
def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data


def valid_transformer(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return valid_data

def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
def data_loader(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=80, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=80)
    return loader
    
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
    for param in model.parameters():
        param.requires_grad = False
    return model
             
def initial_classifier(model, hidden_units):
    if type(hidden_units) == type(None): 
        hidden_units = 4096 #hyperparamters
        print("Default hidden layers = 4096.")

    input_features = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096, bias=True)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(4096, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    return classifier

def choose_system(gpu_arg):
    if not gpu_arg:
        return torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device chosen: ', device)
    return device

# Implement a function for the validation pass
def validation(model, validloader, criterion, device):
    model.to(device)
    valid_loss = 0
    accuracy = 0
    
    for inputs, labels in validloader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def trainer(model, trainloader, validloader, device,
            criterion, optimizer, epochs, print_every, steps):
    if type(epochs) == type(None):
        epochs = 3
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion, device)
                print('Epoch: {0}/{1}'.format(e+1,epochs),
                  'Train loss: {:.2f} -- '.format(running_loss/print_every),
                  'Valid loss: {:.2f} -- '.format(valid_loss/len(validloader)),
                  'Valid accuracy: {:.2f} -- '.format(accuracy/len(validloader))
                     )
                running_loss = 0
                model.train()
    return model
                                                    
def checkpoint(model, save_dir, train_data):
    if type(save_dir) == type(None):
        print('model has no SAVE directory')
    else:
        model.class_to_idx = train_data.class_to_idx
        # checkpoint dictionary
        checkpoint = {'classifier': model.classifier,
                      'architecture': model.name,
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx}
        torch.save(checkpoint, 'checkpoint.pth')

def main():
    args = arg_parser()
    
    data_dir = '/home/workspace/ImageClassifier/flowers/'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = train_transformer(train_dir)
    valid_data = valid_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    trainloader = data_loader(train_data, train=True)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    model = model_forward(args.arch)
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    device = choose_system(gpu_arg=args.gpu)
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    epochs = 3
    steps = 0
    print_every = 50

    trained_model = trainer(model, trainloader, validloader, device,
            criterion, optimizer, args.epochs, print_every, steps)
    
    checkpoint(model, args.save_dir, train_data)
    
if __name__ == '__main__':
    main()
             


    
    