#!/usr/bin/env python
# coding: utf-8



import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import math
torch.cuda.set_device(0)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine



input_size = 784
batch_size = 512
num_epochs = 500
learning_rate = 0.01
hidden_size = 500
number_H =5
probability = 0.0
Weight_Decay_L2 = 0.0
Weight_Decay_L1 = 0.0
random_seed = 42

epoch_interval = 499

Momentum = 0.95
Alpha = 0.0
beta1 = 0.9
beta2 = 0.999

address =''
name = 'lr_'+'batch_size:'+str(batch_size)+'_learning_rate:'+ str(learning_rate) +'_hidden_size:' + str(hidden_size) + '_random_seed :'+ str(random_seed )
address_1 = address+name+'_result.txt'
file1 = open(address_1,'w')


def seed_torch(seed=random_seed):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   
seed_torch()

def cos_similarity_matrix_row(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] = 1 - cosine(matrix[i], matrix[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def cos_similarity_matrix_column(matrix):
    num_column = matrix.shape[1]
    similarity_matrix = np.zeros((num_column, num_column))
    for i in range(num_column):
        for j in range(i, num_column):
            similarity_matrix[i, j] = 1 - cosine(matrix[:,i], matrix[:,j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def Mean(matrix):
    number = matrix.shape[0]
    matrix_mean = matrix.mean()
    mean_out = abs((matrix_mean - (1/number))*(number/(number-1)))
    return mean_out
def Gram_matrix_row(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix,matrix_transpose)
    return Gram_matrix
def Gram_matrix_column(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix_transpose,matrix)
    return Gram_matrix
def L1_regularization(loss):
    l1_regularization = torch.tensor(0.).cuda()
    for param in model.parameters():
        l1_regularization += torch.norm(param, p=1)
    loss += Weight_Decay_L1 * l1_regularization
    return loss
def cos_similarity(name):
    print(f"Parameter name: {name}")
    file1.writelines(f"{name}"+'  ')
    print(f"Parameter value: {param.data.size()}")  
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data)
    cos_sim_column = cos_similarity_matrix_column(param.cpu().data) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(str(mean_cos_sim_row)+'  '+ str(mean_cos_sim_column)+'  ')
    print('='*50)


train_datasets = dsets.MNIST(root = './Datasets', train = True, download = True, transform = transforms.ToTensor())
test_datasets = dsets.MNIST(root = './Datasets', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False)

class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        #self.r = nn.GELU()
        #self.r = nn.Sigmoid()
        #self.r = nn.Tanh()
        self.dropout = nn.Dropout(probability)
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(probability)
        self.norm = nn.BatchNorm1d(hidden)

    def forward(self, x):
        tensor = torch.tensor((),dtype = torch.float32)
        x = self.linear(x)
        x = self.r(x)
        
        for i in  range(number_H):
            if i == 10:
                x = self.dropout(x)
                x = self.linearH[i](x)
                x = self.r(x)
            else:
                x = self.linearH[i](x)
                x = self.r(x) 

        out = self.out(x)
        return out


if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)
criterion = nn.CrossEntropyLoss()


#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = Momentum)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,betas=(beta1, beta2),eps = 1e-8)
optimizer.zero_grad()


for name, param in model.named_parameters():
    print(name)
for name, param in model.named_parameters():

    if name == 'linearH.0.weight':
        cos_similarity(name)
    if name == 'linearH.1.weight':
        cos_similarity(name)
    if name == 'linearH.2.weight':
        cos_similarity(name)
    if name == 'linearH.3.weight':
        cos_similarity(name)        
    if name == 'linearH.4.weight':
        cos_similarity(name)    
        file1.writelines('\n') 

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i,(features, targets) in enumerate(data_loader):
        #features = features.to(device)
        features = Variable(features.view(-1, 28*28)).cuda()
        
        #targets = targets.to(device)
        targets = Variable(targets).cuda()
        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
        outputs= model(images)
        optimizer.zero_grad() 
        loss = criterion(outputs, labels)
        
        L1_regularization(loss)
        
        loss.backward()
        optimizer.step()
        loss_out=round(loss.item(),4)
        if (i+1) % 40 == 0:
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_datasets)//batch_size,loss.item()))
    text_accuracy = round(compute_accuracy(model, test_loader).item(),4)
    print(str(text_accuracy)+'  '+str(loss_out))
    file1.writelines(str(text_accuracy)+'  '+str(loss_out)+'\n')
    if epoch == num_epochs-1:
        file1.writelines('epoch:'+str(epoch)+'  ')
        for name, param in model.named_parameters():   
            if name == 'linearH.0.weight':
                cos_similarity(name)
            if name == 'linearH.1.weight':
                cos_similarity(name)
            if name == 'linearH.2.weight':
                cos_similarity(name)
            if name == 'linearH.3.weight':
                cos_similarity(name)        
            if name == 'linearH.4.weight':
                cos_similarity(name)
                file1.writelines('\n')

file1.close() 







