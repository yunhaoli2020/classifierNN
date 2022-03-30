import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.
	
	Network architecture:
	- Input layer
	- First hidden layer: fully connected layer of size 128 nodes
	- Second hidden layer: fully connected layer of size 64 nodes
	- Output layer: a linear layer with one node per class (in this case 10)

	Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.in1 = nn.Linear(28*28, 128)
        self.FC1 = nn.Linear(128,64)
        self.FC2 = nn.Linear(64,32)
        self.FC3 = nn.Linear(32,10)

    def forward(self, input):
        #raise NotImplementedError()
        y = F.relu(self.in1(input))
        y = F.relu(self.FC1(y))
        y = F.relu(self.FC2(y))
        y = self.FC3(y)
        return y

    

















