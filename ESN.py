import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import func 
import os





data_size=10000#样本大小
device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

#if dtype == torch.double:
#    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float64)
#elif dtype == torch.float:
#    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float32)



t=np.random.rand(data_size)*1400
t=np.sort(t)

tt=np.random.rand(data_size)*1400
tt=np.sort(tt)
ans=func.NARMA(t,data_size)
anss=func.NARMA(tt,data_size)
data=np.zeros([data_size,4])
for i in range (data_size):
	data[i][0]=t[i]
	data[i][1]=ans[i]
	data[i][2]=tt[i]
	data[i][3]=anss[i]

X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
X_data = torch.from_numpy(X_data).to(device)
Y_data = torch.from_numpy(Y_data).to(device)

trX = X_data
trY = Y_data

tsX = np.expand_dims(data[:, [2]], axis=1)
tsY = np.expand_dims(data[:, [3]], axis=1)
tsX = torch.from_numpy(tsX).to(device)
tsY = torch.from_numpy(tsY).to(device)

washout = [2000]
input_size = output_size = 1
hidden_size = 500
loss_fcn = torch.nn.MSELoss()

if __name__ == "__main__":
    start = time.time()
    output=np.zeros([data_size-washout[0],1,1])
    output=torch.from_numpy(output).to(device)
    while(loss_fcn(output, trY[washout[0]:]).item()>0.006):
	    # Training
	    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

	    model = ESN(input_size, hidden_size, output_size)
	    model.to(device)

	    model(trX, washout, None, trY_flat)
	    model.fit()
	    output, hidden = model(trX, washout)
    xlabel=t.reshape(data_size).tolist()[washout[0]:]
    ylabel1=output.reshape(data_size-washout[0]).tolist()
    ylabel0=ans.reshape(data_size).tolist()[washout[0]:]
    plt.plot(xlabel,ylabel0,'b')
    plt.plot(xlabel,ylabel1,'r')
    plt.show()
    print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

    # Test
    output, hidden = model(tsX, [0], hidden)
    print("Test error:", loss_fcn(output[washout[0]:], tsY[washout[0]:]).item())
    print("Ended in", time.time() - start, "seconds.")

#画图
    plt.figure()
    xlabel=tt.reshape(data_size).tolist()[washout[0]:]
    ylabel1=output.reshape(data_size).tolist()[washout[0]:]
    ylabel0=anss.reshape(data_size).tolist()[washout[0]:]
    plt.plot(xlabel,ylabel0,'b')
    plt.plot(xlabel,ylabel1,'r')
    plt.show()
