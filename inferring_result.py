import torch
import cv2
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
                
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(6 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 70)
        self.fc3 = nn.Linear(70,2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))        
        x = self.pool(F.relu(self.conv2(x)))        
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


PATH = './figures_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

fname = 'pic.png'

img_size = 64
# чтение картинки в чб
img_array = cv2.imread(os.path.join('.', fname), cv2.IMREAD_GRAYSCALE)

# масштабирование картинки под нейронку
new_array = cv2.resize(img_array, (img_size, img_size))
# из списка матриц пикселей картинки получаем матрицу списков пикселей картинок
X = []
for features in new_array:
    X.append(features)

X = np.array(X).reshape(1, 1, img_size, img_size)

# определение отклика
tensorX = torch.Tensor(X[:1])
outputs = net(tensorX)
_, predicted = torch.max(outputs, 1)
classes = ['circle','polygon']
print(classes[predicted[0]])