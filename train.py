import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import func 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_size=2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n=data_size



#定义网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,4,kernel_size=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc0= nn.Linear(1,500)
        self.fc1= nn.Linear(30,1)
    def forward(self, x):
        out = self.fc0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc1(out)
        return out
#生成训练数据集
t=np.random.rand(data_size)*1400
t=np.sort(t)
m=np.size(t)
ans=func.NARMA(t,m)
plt.plot(xlabel,ans)
ans=torch.from_numpy(ans).float()
ans=ans.reshape(data_size,1,1)
t=torch.from_numpy(t).float()
t=t.reshape(data_size,1,1)
t=t.to(device)
ans=ans.to(device)
#网络训练

net = Net() 
net=net.to(device)
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.002)
loss_func = torch.nn.MSELoss() 
Num=2000#迭代次数
net.train()
for i in range(Num):
    net.train()
    out = net(t)               # input x and predict based on x
    loss = loss_func(out,ans)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    prediction = torch.max(out, 1)[1]
    print(loss)
torch.save(net.state_dict(),'Mynet.pkl')
#画图
xlabel=t.reshape(data_size).tolist()
out=out.cpu()
test_out=out.reshape(n).detach().numpy().tolist()
plt.plot(xlabel,test_out)


plt.show()

