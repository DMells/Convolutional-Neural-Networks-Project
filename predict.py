# The second file, predict.py, uses a trained network to predict the class for an input image.
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

predict_parser = argparse.ArgumentParser(
    description='This is David Mellors project submission for the AIPND')
 # User should be able to type python predict.py input checkpoint
    # Non-optional image file input
predict_parser.add_argument('input', action="store", nargs='*', default='/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg')
    # Non-optional checkpoint 
predict_parser.add_argument('checkpoint', action="store", nargs='*', default='/home/workspace/paind-project/checkpoint.pth')
    # Choose top K 
predict_parser.add_argument('--top_k', action="store", dest="top_k", type=int, default=3)
    # Choose category list
predict_parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')
  # Choose processor
predict_parser.add_argument('--processor', action="store", dest="processor", default="GPU")
    

predict_args = predict_parser.parse_args()
print("Image input: ", predict_args.input, "Checkpoint: ", predict_args.checkpoint, "TopK: ", predict_args.top_k, "Category names: ", predict_args.category_names, "Processor: ", predict_args.processor)
print(predict_args)

from ldngandpreprop import *

def main():
    
    load_checkpoint(predict_args.checkpoint)
#     load_checkpoint('/home/workspace/paind-project/checkpoint.pth')
   
    image = process_image(predict_args.input)
    imshow(image)
    problist, prenadj = predict(image, model, predict_args.top_k, predict_args.processor)
    
    import json
    
    with open(predict_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
#     # TODO: Display an image along with the top 5 classes
    imshow(img)   
    list(cat_to_name.items())[0][1]
    names = []
    for i in prednadj:

        j = list(cat_to_name.items())[i][1]
        names.append(j)

    print(names)
    print(problist)

    result = list(zip(names,problist))
    print(result)    


    base_color = sb.color_palette()[0]
    sb.barplot(y = names, x = problist, color = base_color)
    
if __name__ == "__main__":
    main()
