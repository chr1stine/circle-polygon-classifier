import torch
import cv2
import os
import numpy as np
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")
sigm = torch.nn.Sigmoid()
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()                
        self.conv1 = torch.nn.Conv2d(1, 3, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(3, 6, 3)
        self.conv3 = torch.nn.Conv2d(6,12,3)
        self.fc1 = torch.nn.Linear(12 * 14 * 14, 1)
        self.dout = torch.nn.Dropout(0.5)    
    def forward(self, x):
        x = self.dout(self.pool(torch.nn.functional.relu(self.conv1(x))))
        x = self.dout(self.pool(torch.nn.functional.relu(self.conv2(x))))
        x = self.dout(self.pool(torch.nn.functional.relu(self.conv3(x))))
        x = torch.flatten(x,1)
        x = sigm(self.fc1(x))
        return x
PATH = './figures_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
net.eval()
fname = 'pic.png'
img_size = 128
img_array = cv2.imread(os.path.join('.', fname), cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (img_size, img_size))
X = []
for features in new_array:
    X.append(features)
X = np.array(X).reshape(img_size, img_size,1)
tr = transforms.Compose([transforms.ToPILImage(),transforms.GaussianBlur(9,9),transforms.ToTensor()])
tX = tr(X + np.array(torch.rand(img_size,img_size,1))).unsqueeze(1)
output = net(tX).squeeze()
classes = ['circle','square']
predicted = torch.heaviside(output-0.4,torch.tensor([1.])).int()
print(classes[predicted])
