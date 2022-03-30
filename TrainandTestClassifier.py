import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.my_dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from models.run_model import run_model
from models.network import Digit_Classifier
from models.SVM import SVMClassifier
from data.load_data import load_mnist_data
import time
import matplotlib.pyplot as plt


# The directory to save trained network models
PATH = "train_nn.pt"

#Generate training and testing data, including images and targets
train_features,test_features,train_targets,test_targets = load_mnist_data(10, fraction= 0.8)

print(train_features.shape)
print(train_targets.shape)
print(test_features.shape)
print(test_targets.shape)

# Take 16 images in training dataset to visualize
trainImages = train_features[0:16,:]
testImages = test_features[0:16,:]
trainImages = np.reshape(trainImages, (16 ,28,28))
testImages = np.reshape(testImages, (16, 28, 28))
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(trainImages[i,:,:])
    plt.axis('off')

plt.show()

# Create training and testing dataloader
train_dataset = MyDataset(train_features, train_targets)
test_dataset = MyDataset(test_features,test_targets)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle= True)

#Train and test SVM classifier first
start = time.time()
betaMat, vMat = SVMClassifier(train_features, train_targets, test_features, test_targets)

vMatTotal = np.reshape(vMat, (1, vMat.shape[0]))
vMatTotal = np.tile(vMat, (test_features.shape[0], 1))

predictTotal = np.matmul(test_features, betaMat) - vMatTotal
predictTotalTarget = np.argmax(predictTotal, axis=1)
print(predictTotal.shape)
print(predictTotalTarget.shape)
errorTotal = 0
for i in range(test_targets.shape[0]):
    if predictTotalTarget[i] != test_targets[i]:
        errorTotal += 1
accuracyTotal = 1 - errorTotal / test_targets.shape[0]
end = time.time()
print("Testing accuracy of SVM classifier is ", accuracyTotal)
print("training time is ", end-start)

# Initialize model, optimizer and train the model
train_model = Digit_Classifier()
optimizer = optim.SGD(train_model.parameters(), lr=0.01)
start = time.time()
train_model,train_loss,train_acc = run_model(train_model, running_mode='train', train_set= train_dataset, batch_size=10, n_epochs= 100)

# Test the model using tetsting dataset
test_loss,test_acc = run_model(train_model, running_mode= 'test', test_set= test_dataset)
end = time.time()
print("Testing accuracy of NN classifier is ", test_acc / 100)
print("training time is ", end-start)

# Save network model (for running handwritten classifier)
torch.save(train_model.state_dict(), PATH)
