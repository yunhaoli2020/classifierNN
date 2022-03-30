import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None, 
	batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
	"""
	This function either trains or evaluates a model. 

	training mode: the model is trained and evaluated on a validation set, if provided. 
				   If no validation set is provided, the training is performed for a fixed 
				   number of epochs. 
				   Otherwise, the model should be evaluted on the validation set 
				   at the end of each epoch and the training should be stopped based on one
				   of these two conditions (whichever happens first): 
				   1. The validation loss stops improving. 
				   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs: 

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset 
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
	learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model 
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model 
    loss: dictionary with keys 'train' and 'valid'
    	  The value of each key is a list of loss values. Each loss value is the average
    	  of training/validation loss over one epoch.
    	  If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
    	 The value of each key is a list of accuracies (percentage of correctly classified
    	 samples in the dataset). Each accuracy value is the average of training/validation 
    	 accuracies over one epoch. 
    	 If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set. 
    accuracy: percentage of correctly classified samples in the testing set. 
	

	"""

	if running_mode == 'train':
		train_loader = DataLoader(train_set, batch_size= batch_size, shuffle= shuffle)
		if valid_set is not None:
			valid_loader = DataLoader(valid_set,batch_size= batch_size, shuffle=shuffle)
		
		optimizer = optim.SGD(model.parameters(), lr= learning_rate)
		loss = {'train':[], 'valid':[]}
		acc = {'train':[], 'valid':[]}

		valid_loss = 1000
		train_loss_mat = np.array([])
		train_acc_mat = np.array([])
		valid_loss_mat = np.array([])
		valid_acc_mat = np.array([])

		for i in range(n_epochs):
			model,train_loss,train_accuracy = _train(model, train_loader, optimizer)
			if valid_set is not None:
				valid_loss, valid_accuracy = _test(model, valid_loader, optimizer)
				valid_loss_mat = np.append(valid_loss_mat, valid_loss)
				valid_acc_mat = np.append(valid_acc_mat, valid_accuracy)
			train_loss_mat = np.append(train_loss_mat, train_loss)
			train_acc_mat = np.append(train_acc_mat, train_accuracy)
			
			if valid_loss < stop_thr:
				break
				
		print("current epoch is ", i)
		loss['train'] = train_loss_mat
		loss['valid'] = valid_loss_mat
		acc['train'] = train_acc_mat
		acc['valid'] = valid_acc_mat
		return model,loss,acc
	if running_mode == 'test':
		test_loader = DataLoader(test_set, batch_size= batch_size, shuffle= shuffle)
		test_loss, test_accuracy = _test(model, test_loader)
		return test_loss,test_accuracy


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

	"""
	This function trains the proposed network model on a given dataset for only one epoch.

	Inputs:
	model: the neural network to be trained
	data_loader: for loading the netowrk input and targets from the training dataset
	optimizer: the optimiztion method, e.g., SGD 
	device: we run everything on CPU, GPU and CUDA is not required.

	Outputs:
	model: the trained model
	train_loss: average training loss
	train_accuracy: average training accuracy on training dataset
	"""

	train_loss = 0
	i = 0
	loss_function = nn.CrossEntropyLoss()
	predict_mat = []
	target_mat = []
	for step, (data, target) in enumerate(data_loader):
		optimizer.zero_grad()
		output = model(data.float())
		loss = loss_function(output, target.long())
		train_loss = train_loss + loss.detach().numpy()
		_, predicted = torch.max(output.data, 1)
		predicted = torch.max(output, 1)[1].data.numpy().squeeze()
		predicted = predicted.tolist()
		predict_mat.append(predicted)
		target_mat.append(target.numpy().squeeze().tolist())

		loss.backward()
		optimizer.step()
		i += 1
	train_loss = train_loss / i
	train_accuracy = 100 * np.mean(np.array(predict_mat) == np.array(target_mat))
	return model, train_loss,train_accuracy




def _test(model, data_loader, device=torch.device('cpu')):
	"""
	This function evaluates the trained neural network on the testing dataset

	Inputs:
	model: trained neural network model
	data_loader: for loading the netowrk input and targets from the testing dataset
	device: CPU only, same as training function.

	Output:
	test_loss: mean loss on testing set
	test_accuracy: testing accuracy, represented as the numer of correctly classified images
	"""
	
	test_loss = 0
	i = 0
	loss_function = nn.CrossEntropyLoss()
	predict_mat = []
	target_mat = []
	for step, (data, target) in enumerate(data_loader):
		
		output = model(data.float())
		loss = loss_function(output, target.long())
		test_loss = test_loss + loss.detach().numpy()
		predicted = torch.max(output, 1)[1].data.numpy().squeeze()
		predicted = predicted.tolist()
		predict_mat.append(predicted)
		target_mat.append(target.numpy().squeeze().tolist())
		i += 1
	test_loss = test_loss / i
	test_accuracy = 100 * np.mean(np.array(predict_mat) == np.array(target_mat))
	return test_loss, test_accuracy
	



