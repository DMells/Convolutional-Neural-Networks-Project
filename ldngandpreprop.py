# This file loads and preprocesses the imagess
from train import main
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import argparse
from predict import predict_args
from modelfns import *

def load_checkpoint(filepath):
       
        model = torch.load(filepath)

        epochs = checkpoint['epochs']
        model.load_state_dict()
        model.class_to_idx = checkpoint['image_datasets']
        optimizer.load_state_dict(model['optimizer'])
        return model

    
import PIL 
from PIL import Image


def process_image(image):
    
    img = Image.open(image)

    width, height = img.size
    print(img.size)
    
    if width < height:
        img.thumbnail((256,height))
    else:
        img.thumbnail((width,256))

    print(img.size)

# To crop the image:1
    imgwidth = img.size[0]
    imgheight = img.size[1]
    halfimgwidth = imgwidth//2
    halfimgheight = imgheight//2
    
    # // is used for integer division (floor)
    crop_square = (imgwidth//2 - 112, 
                   imgheight//2 - 112, 
                   imgwidth//2 + 112, 
                   imgheight//2 + 112)


    img = img.crop(crop_square)
    print(img)
    #convert to tensor to normalise
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    print(img)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    img = normalize(img)
    print(img.shape)
    img = np.array(img)

    print(img.shape)

    # Transpose image to get dimensions in correct order for pytorch
    img = np.ndarray.transpose(img)
    print(img.shape)
    return img

# To test the function:
# img = (data_dir + '/test' + '/1/' + '/image_06743.jpg')
# img = (data_dir + '/test' + '/1/' + '/image_06752.jpg')
#Run function
import numpy

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)

    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.class_to_idx = train_data.class_to_idx
    ctx = model.class_to_idx
    to_tensor=transforms.ToTensor()
    image_path = to_tensor(image_path)
    image_path = image_path.unsqueeze(0)
    
     if processor == "GPU":
        output = model.forward(Variable(image_path.cuda(), volatile=True))
    else:
        output = model.forward(Variable(image_path.cpu(), volatile=True))
    
    ps = torch.exp(output)
    k_prob, v_predn = ps.topk(5)
    
    #To pull off probabilities into separate list
    problist = []
    for i in k_prob[0].data:
        problist.append(i)
    print(problist)
    
    
    #To pull off unadjusted predicted values into separate list
    prednlist = []
    # To pull off adjusted predicted values from ctx to give correct indices
    prednadj = []
    print("v_predn" ,v_predn)
    for i in v_predn[0].data:
        prednlist.append(i)
        prednadj.append(ctx[str(i)])
    print("prednlist" ,prednlist)
    print("prednadj", prednadj)
    
    return problist, prednadj





