import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

#定义函数
y0=10
y1=12
d0=0.3
d1=0.6
d2=2.0
d3=1.3
def fun_I(t):	
	y=-1/490000*pow(t,2)+1400/490000*t
	return(y)
def fun_U(t):
	if t>=0 and t<300:	
		y=0.125
	if t>=300 and t<500:
		y=0.25
	if t>=500 and t<900:
		y=0.5
	if t>=900 and t<1100:
		y=0.25
	if t>=1100:
		y=0.125
	return(y)
def NARMA(t,m):
	u=np.zeros(m)
	y=np.zeros(m)	
	I=fun_I(t)
	y[0]=y0
	y[1]=y1
	for i in range (m):
		u[i]=fun_U(t[i])
	for i in range (m):
		if t[i]<=0:
			y[i]=0
		y[i]=(1-d1)*y[i-1]+(1-d3)*y[i-1]*u[i-1]/u[i-2]+(d3-1)*(1+d1)*y[i-2]*u[i-1]/u[i-2]+d0*d2*u[i-1]*I[i-2]-d0*u[i-2]*y[i-1]+d0*(1+d1)*u[i-1]*y[i-2]
	return y
#定义网络
class Net(torch.nn.Module):
    def __init__(self, num_classes=1):
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
        self.fc3= nn.Linear(11,num_classes)
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
ans=NARMA(t,m)
ans=torch.from_numpy(ans).float()
ans=ans.reshape(2000,1,1)
t=torch.from_numpy(t).float()
t=t.reshape(2000,1,1)
#生成测试数据集
tt=np.random.rand(400)*1400
tt=np.sort(tt)

#tt=np.hstack((p1,p2))
#tt=np.sort(tt)
n=np.size(tt)

anss=NARMA(tt,n)
anss=torch.from_numpy(anss).float()

anss=anss.reshape(400,1,1)
tt=torch.from_numpy(tt).float()
tt=tt.reshape(400,1,1)

#dataset=[torch.from_numpy(t).float(),torch.from_numpy(ans).float()]

#网络训练
num_classes=1
net = Net(num_classes) 

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
plt.show()
net.eval()
test_out=net(tt)
loss = loss_func(test_out,anss)
print(loss)

