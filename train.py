import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import func 
#定义网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc0= nn.Linear(1,200)
        self.fc1= nn.Linear(499,3000)
        self.fc2= nn.Linear(749, 2000)
        self.fc3= nn.Linear(11,1)
    def forward(self, x):
        out = self.fc0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        #out = self.fc1(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.reshape(out.size(1), -1)
        #out = self.fc2(out)
        out = self.fc3(out)
        return out
#生成训练数据集
t=np.random.rand(2000)*1400
t=np.sort(t)
m=np.size(t)
ans=func.NARMA(t,m)
ans=torch.from_numpy(ans).float()
ans=ans.reshape(2000,1,1)
t=torch.from_numpy(t).float()
t=t.reshape(2000,1,1)
#网络训练

net = Net() 

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.MSELoss() 
Num=200#迭代次数
net.train()
for i in range(Num):
    net.train()
    out = net(t)               # input x and predict based on x
    loss = loss_func(out,ans)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    prediction = torch.max(out, 1)[1]
    gg=loss.detach().numpy()
    plt.scatter(i,gg)
torch.save(net.state_dict(),'Mynet.pkl')
plt.show()
