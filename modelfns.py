# This is where the models functions and classes are stored
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

# Define transforms for the training, validation, and testing sets
def trainer(model, data_dir, save_dir, learning_rate, num_epochs, hidden_units, processor):
    if model == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("No model specified")
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])])

    validn_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                                (0.229, 0.224, 0.225))])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,
                                    transform=train_transforms)

    validn_data = datasets.ImageFolder(valid_dir,
                                    transform=validn_transforms)

    # Removed test_data parts as not testing it here. Only training, then validating.
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validnloader = torch.utils.data.DataLoader(validn_data, batch_size=32, shuffle=True)


    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # if user input was vgg :
    classifier = nn.Sequential(OrderedDict([            
        # To capture the original image sizes ( 3 color channels x w x h) and flatten them into a 1D matrix. 
        #     This is then the input layer size
                              ('fc1', nn.Linear(25088, hidden_units)), 
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 1000)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(1000, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    #Verify availability of GPU
    use_gpu = torch.cuda.is_available()
    print("GPU available:", use_gpu)

    #Train network-1
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if processor == "GPU":
        model.cuda()
    else:
        model.cpu()

    #Train network-2
    epochs = int(num_epochs)
    steps = 0
    training_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        for images, labels in iter(trainloader):
            steps += 1
            inputs = Variable(images.cuda())
            targets = Variable(labels.cuda())
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.data[0]

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                validn_loss = 0
                for ii, (images, labels) in enumerate(validnloader):
                    if processor == "GPU":
                        inputs = Variable(images.cuda(), volatile=True)
                        labels = Variable(labels.cuda(), volatile=True)
                    else:
                        inputs = Variable(images.cpu(), volatile=True)
                        labels = Variable(labels.cpu(), volatile=True)
                     

                    output = model.forward(inputs)
                    validn_loss += criterion(output, labels).data[0]
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()


                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(training_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validn_loss/len(validnloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validnloader)))

                training_loss = 0

                # Make sure dropout is on for training
                model.train()
    print("Training complete")

    # Save the checkpoint 
#     print("Model: \n\n", model, '\n')
#     print("The state dict keys: \n\n", model.state_dict().keys())
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
                  'state_dict': model.state_dict(),
                  'image_datasets' : model.class_to_idx,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'model': model,
                 }

    torch.save(checkpoint, save_dir)
    print("Checkpoint saved!")

    return model
