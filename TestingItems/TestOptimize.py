#!/usr/bin/python3

from fastai.basics import tensor, nn
import torch, numpy, pandas

def hypothesis(vertLatents,horiLatents):
    # return the value of the latent vectors' inner/dot product
    return torch.matmul(vertLatents,horiLatents)

def mse(horiLatents,vertLatents,y):
    # calculate RMSE where errorVec is a matrix of errors
    #errorVec = numpy.subtract(y_hat,y)
    return ((torch.matmul(horiLatents,vertLatents)-y)**2).mean()

def update(horiLatents,vertLatents):
    # perform gradient descent
    loss = mse(vertLatents,horiLatents,blockData)
    loss.backward()
    if t%100 == 0:
        print(t,'-------------',loss)
        dict1 = {'vertLatents':[vertLatents.grad,vertLatents.is_leaf,
                                vertLatents.requires_grad],
                 'horiLatents':[horiLatents.grad,horiLatents.is_leaf,
                                horiLatents.requires_grad],}
        df1 = pandas.DataFrame.from_dict(dict1,orient='index')
        df1.columns = ['grad','is_leaf','requires_grad']
        #print(df1)
    with torch.no_grad():
        vertLatents.sub_(lr * vertLatents.grad)
        vertLatents.grad.zero_()
        horiLatents.sub_(lr * horiLatents.grad)
        horiLatents.grad.zero_()
    return loss.item()

vecLatents = 10
shape = (20,14)
# random large block of data
blockData = tensor(numpy.random.random_sample(shape))

horiLatents = \
        nn.Parameter(tensor(numpy.random.random_sample((vecLatents,shape[1]))))
vertLatents = \
        nn.Parameter(tensor(numpy.random.random_sample((shape[0],vecLatents))))

lr = 1e-1
horiLatents.requires_grad_(True)
vertLatents.requires_grad_(True)
lossDict = {}
for t in range(10001):
    lossDict[t] = update(horiLatents,vertLatents)
pandas.DataFrame.from_dict(lossDict,orient='index').to_csv('lossDict.csv',
                                                           index=False)
pandas.DataFrame(horiLatents.data.tolist()).to_csv('horiLatents.csv',
                                                   index=False)
pandas.DataFrame(vertLatents.data.tolist()).to_csv('vertLatents.csv',
                                                   index=False)
pandas.DataFrame(blockData.data.tolist()).to_csv('blockData.csv',
                                                   index=False)
pandas.DataFrame(torch.matmul(vertLatents,horiLatents).data.tolist()).to_csv(
    'optimized.csv',index=False)
