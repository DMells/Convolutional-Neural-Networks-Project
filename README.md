# Convolutional-Neural-Networks-Project

This project involves two main parts:

1. An image classifier model, which uses pre-trained neural networks (VGG) to train a neural network to recognise any image of
a flower that I wish to pass to it, and the resulting output is a prediction by way of a list of probabilities and names
of exactly what type of flower it is.

2. A program built using Python which allows any user to input a series of arguments via arg_parser in order to both train their own model and also to use my trained model to make a prediction of the 5 most likely flowers that the image represents.

The output of the trained model can be seen in the .html file in this repository.

The model can be trained using your own set of training data by calling the train.py file in your command line application, and you can make a prediction of your own flower image by calling the predict.py file. The other two .py files are for separation of concerns only and pull through automatically into the aforementioned files.

David Mellor, May 2018
