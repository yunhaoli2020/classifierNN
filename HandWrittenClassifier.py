import torch
import numpy as np
from data.my_dataset import MyDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from drawingtest import Drawing
from models.network import Digit_Classifier



# Now let's try to draw your own digit and test the handwritten digit classifier!

# Draw a number using GUI and mouse, transfer the image to the format of MNIST dataset.
writtenImage = Drawing()
plt.imshow(writtenImage)
plt.show()
writtenImage = np.reshape(writtenImage, (1, writtenImage.shape[0] * writtenImage.shape[0]))
writtenTarget = np.array([0])
writtenDataset = MyDataset(writtenImage, writtenTarget)
writtenLoader = DataLoader(writtenDataset, batch_size=1, shuffle=False)

# Load trained nn classifier
train_model = Digit_Classifier()
train_model.load_state_dict(torch.load("train_nn.pt"))
train_model.eval()
for step, (data, target) in enumerate(writtenLoader):
    output = train_model(data.float())
    predicted = torch.max(output, 1)[1].data.numpy().squeeze()
print("You are writing number ", predicted)