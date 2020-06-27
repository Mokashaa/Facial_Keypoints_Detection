## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # Covolutional Layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        #self.conv4 = nn.Conv2d(128, 256, 2)
        
        # Maxpooling Layer
        self.pool  = nn.MaxPool2d(2, stride=2)
        
        # Dropouts
        self.Droupout1 = nn.Dropout(0.1)
        self.Droupout2 = nn.Dropout(0.2)
        self.Droupout3 = nn.Dropout(0.3)
        self.Droupout4 = nn.Dropout(0.4)
        self.Droupout5 = nn.Dropout(0.5)
        self.Droupout6 = nn.Dropout(0.6)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(26*26*128, 512)
        self.fc2 = nn.Linear(512, 136)
        #self.fc3 = nn.Linear(512, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        #input size = 224x224x1
        
        # First - Convolution + Activation + Pooling + Dropout
        x = self.Droupout1(self.pool(F.relu(self.conv1(x))))
        #size after conv = 220x220x32 
        #size after pooling = 110x110x32
        
        
        # Second - Convolution + Activation + Pooling + Dropout
        x = self.Droupout2(self.pool(F.relu(self.conv2(x))))
        #size after conv = 108x108x64 
        #size after pooling = 54x54x64
        
        
        # Third - Convolution + Activation + Pooling + Dropout
        x = self.Droupout3(self.pool(F.relu(self.conv3(x))))
        #size after conv = 52x52x128 
        #size after pooling = 26x26x128
        
        
        # Fourth - Convolution + Activation + Pooling + Dropout
        #x = self.Droupout4(self.pool(F.relu(self.conv4(x))))
        #size after conv = 25x25x256 
        #size after pooling = 12*12*256
        
        
        #Flatten
        x = x.view(x.size(0), -1)
        
        # First - Dense + Activation + Dropout
        x = self.Droupout5(F.relu(self.fc1(x)))
        # Second - Dense + Activation + Dropout
        x = self.fc2(x)
        
        #Final Layer
        #x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
