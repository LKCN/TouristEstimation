import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from torch import nn
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
import math
import time

#数据加载 
print('读取原始训练数据:')
delay=1.5
time.sleep(delay)
trainData=pd.read_excel('train.xlsx',header=0)

#数据分配和数据归一化
ss=MinMaxScaler()
dataTrain=torch.from_numpy((trainData.values).astype(np.float32))
train=dataTrain[:,0:6].clone().detach()
train=ss.fit_transform(train)
train=torch.from_numpy(train)
train=train.reshape(-1,1,6)

trainLabel=dataTrain[:,6:7].clone().detach()
trainLabel=ss.fit_transform(trainLabel)
trainLabel=torch.from_numpy(trainLabel)
trainLabel=trainLabel.reshape(-1,1,1)
print(dataTrain)
print('归一化后训练数据:')
time.sleep(delay)
print(ss.fit_transform(dataTrain))
#pytorch加载训练数据集
trainDataset=Data.TensorDataset(train,trainLabel)
trainLoader = Data.DataLoader(dataset = trainDataset, batch_size = 126, shuffle = True)

#搭建网络
class lstm_reg(nn.Module):
    def __init__(self,input_size,hidden_size, output_size=1,num_layers=2):
        super(lstm_reg,self).__init__()
 
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers)
        self.reg = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x, _ = self.rnn(x)
        s,b,h = x.shape
        x = x.view(s*b, h)
        x = torch.sigmoid(self.reg(x))
        x = x.view(s,b,-1)
        return x

if torch.cuda.is_available():
    model = lstm_reg(6,30).cuda()
else:
    model = lstm_reg(6,30)
#损失函数与优化器 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#开始训练
print('开始训练:')
num_epochs = 3000
for epoch in range(num_epochs):
    for step,(batch_x,batch_y) in enumerate(trainLoader):
        
        if torch.cuda.is_available():

            inputs = Variable(batch_x).cuda()
            inputs.float()
            target = Variable(batch_y.clone()).cuda()
        else:
            inputs = Variable(batch_x)
            inputs.float()
            target = Variable(batch_y.clone())

        # 向前传播
        out = model(inputs.float())
        #损失计算
        loss = criterion(out, target.float())

        # 向后传播
        optimizer.zero_grad() # 注意每次迭代都需要清零
        loss.backward()
        optimizer.step()

        if (epoch+1) %50 == 0:
            print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.item()))

#测试
model.eval()
print('读取原始测试数据:')
time.sleep(delay)
testData=pd.read_excel('test.xlsx',header=0)
dataTest=torch.from_numpy((testData.values).astype(np.float32))
dataTestLabel=dataTest[:,6:7].clone().detach()

test=dataTest[:,0:6].clone().detach()
test=ss.fit_transform(test)
test=torch.from_numpy(test)
test=test.reshape(-1,1,6)

testOriginalLabel=dataTest[:,6:7].clone().detach()
testLabel=ss.fit_transform(testOriginalLabel)
testLabel=torch.from_numpy(testLabel)
testLabel=testLabel.reshape(-1,1,1)
print(dataTest)
print('归一化后测试数据:')
time.sleep(delay)
print(ss.fit_transform(dataTest))

#pytorch加载训练数据集
testDataset=Data.TensorDataset(test,testLabel)
testLoader = Data.DataLoader(dataset = testDataset, batch_size = 15, shuffle = False)
plt.figure(dpi=120,figsize=(10,8))
anchor = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
tick_labels = ['2019-01-04', '2019-01-17','2019-01-25','2019-02-15','2019-02-25','2019-03-02','2019-03-07','2019-03-21','2019-04-29','2019-05-13','2019-08-01','2019-08-19','2019-08-27','2019-08-31','2019-09-03']
for step,(test_x,test_y) in enumerate(testLoader):

    if torch.cuda.is_available():
        inputs = Variable(test_x).cuda()
        inputs.float()
        target = Variable(test_y.clone()).cuda()
        originalLabel = Variable(testOriginalLabel.clone()).cuda()
    else:
        inputs = Variable(test_x)
        inputs.float()
        target = Variable(test_y.clone())
        originalLabel = Variable(testOriginalLabel.clone())

    predict = model(inputs.float())
    predicttoreal=(predict.view(-1,1))*max(originalLabel.float())+min(originalLabel.float())
    #损失计算
    lossTest = criterion(predict, target.float())
    plt.plot(dataTestLabel.data.numpy(),color='r', label='Original Data')
    plt.plot(predicttoreal.cpu().data.numpy(),color='b', label='Estimated Data')
    plt.xticks(anchor,tick_labels,rotation=60)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 18})
    print('test loss:{:.6f}'.format(lossTest.item()))
    plt.show()