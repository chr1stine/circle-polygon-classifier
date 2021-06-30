import torch
import cv2
import os
import numpy as np
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()                
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.conv3 = nn.Conv2d(6,12,3)
        self.conv4 = nn.Conv2d(12,24,3)
        self.fc1 = nn.Linear(24 * 6 * 6, 25)
        self.fc2 = nn.Linear(25, 5)
        self.fc3 = nn.Linear(5, 1)
        self.dout = nn.Dropout(0.5)    
    def forward(self, x):
        x = self.dout(self.pool(F.relu(self.conv1(x))))
        x = self.dout(self.pool(F.relu(self.conv2(x))))
        x = self.dout(self.pool(F.relu(self.conv3(x))))
        x = self.dout(self.pool(F.relu(self.conv4(x))))
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
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
