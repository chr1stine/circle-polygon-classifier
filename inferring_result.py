import torch
import cv2
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()                
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.conv3 = nn.Conv2d(6,12,3)
        self.fc1 = nn.Linear(12 * 14 * 14, 2)
        self.dout = nn.Dropout(0.2)    
    def forward(self, x):
        x = self.dout(self.pool(self.conv1(x)))
        x = self.dout(self.pool(self.conv2(x)))
        x = self.dout(self.pool(self.conv3(x)))
        x = torch.flatten(x,1)
        x = self.fc1(x)        
        return x
PATH = './figures_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
fname = 'pic.png'
img_size = 128
img_array = cv2.imread(os.path.join('.', fname), cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (img_size, img_size))
X = []
for features in new_array:
    X.append(features)
X = np.array(X).reshape(1, 1, img_size, img_size)
tensorX = torch.Tensor(X[:1])
outputs = net(tensorX)
_, predicted = torch.max(outputs, 1)
classes = ['circle','polygon']
print(classes[predicted[0]])