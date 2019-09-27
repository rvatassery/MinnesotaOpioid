#!/usr/bin/python3

import os, sys, re, inspect, pandas, glob, numpy, torch

from fastai.basics import tensor, nn
from sklearn.metrics.pairwise import cosine_similarity

def hypothesis(countyVec,yearVec):
    # return the value of the latent vectors' inner/dot product
    return torch.matmul(countyVec,yearVec)

def mse(countyVec,yearVec,y):
    # calculate RMSE where errorVec is a matrix of errors
    #errorVec = numpy.subtract(y_hat,y)
    return ((torch.matmul(countyVec,yearVec)-y)**2).mean()

def update(countyVec,yearVec):
    # perform gradient descent
    loss = mse(countyVec,yearVec,opioidTensor)
    loss.backward()
    if t%1000 == 0:
        print(t,'-------------',loss)
        dict1 = {'countyVec':[countyVec.grad,countyVec.is_leaf,
                                countyVec.requires_grad],
                 'yearVec':[yearVec.grad,yearVec.is_leaf,
                                yearVec.requires_grad],}
        df1 = pandas.DataFrame.from_dict(dict1,orient='index')
        df1.columns = ['grad','is_leaf','requires_grad']
        #print(df1)
    with torch.no_grad():
        countyVec.sub_(lr * countyVec.grad)
        countyVec.grad.zero_()
        yearVec.sub_(lr * yearVec.grad)
        yearVec.grad.zero_()
    return loss.item()

###########################################################################

opioidIn = pandas.read_csv('OpioidMNDataByCounty.csv')
opioidInPivot = opioidIn.pivot(index='County',columns='year',
                                values='TotalPrescripts').reset_index()
opioidInPivot[2017] = numpy.where(opioidInPivot.County == 'ST. LOUIS',
                                    54.4,
                                    opioidInPivot[2017])
colYears = opioidInPivot.columns[(opioidInPivot.columns != 'County')]
opioidInPivot.dropna(axis=0,inplace=True)
opioidInPivot.drop(opioidInPivot.index[opioidInPivot.County ==
                                       'SAINT LOUIS'].tolist(),inplace=True)
opioidInPivot.to_csv('OpioidMNDataByCountyPivot.csv',sep='\t',index=False)
colsNotCounty = opioidInPivot.columns.tolist()
colsNotCounty.remove('County')
opioidTensor = tensor(opioidInPivot[colsNotCounty].values)

shape = opioidTensor.shape
vecLatents = 15

###########################################################################

countyVec = \
        nn.Parameter(tensor(numpy.random.random_sample((shape[0],vecLatents))))
yearVec = \
        nn.Parameter(tensor(numpy.random.random_sample((vecLatents,shape[1]))))

lr = 1e-1
countyVec.requires_grad_(True)
yearVec.requires_grad_(True)

lossDict = {}
for t in range(10001):
    lossDict[t] = update(countyVec,yearVec)
    
pandas.DataFrame.from_dict(lossDict,orient='index').to_csv('lossDict.csv',
                                                           index=False)
pandas.DataFrame(yearVec.data.tolist()).to_csv('yearVec.csv',
                                                   index=False)
pandas.DataFrame(countyVec.data.tolist()).to_csv('countyVec.csv',
                                                   index=False)
pandas.DataFrame(torch.matmul(countyVec,yearVec).data.tolist()).to_csv(
    'optimized.csv',index=False)

lenCty       = shape[0]
lenYear      = shape[1]
cosSimCounty = numpy.empty((lenCty,lenCty))
for countyVIter,countyV in zip(range(lenCty),countyVec.data.tolist()):
    for countyHIter,countyH in zip(range(lenCty),countyVec.data.tolist()):
        cosSimCounty[countyHIter,countyVIter] = \
                                        cosine_similarity([countyV],[countyH])

cosSimYear = numpy.empty((lenYear,lenYear))
for yearVIter,yearV in zip(range(lenYear),yearVec.data.tolist()):
    for yearHIter,yearH in zip(range(lenYear),yearVec.data.tolist()):
        cosSimYear[yearHIter,yearVIter] = \
                                        cosine_similarity([yearV],[yearH])

pandas.DataFrame(cosSimCounty,index=opioidInPivot.County,
                 columns=opioidInPivot.County.tolist()).to_csv(
                     'cosSimCounty.csv')
print(cosSimYear.shape,cosSimCounty.shape)
pandas.DataFrame(cosSimYear,columns=opioidInPivot[colsNotCounty].columns,
                 index=opioidInPivot[colsNotCounty].columns).to_csv(
                     'cosSimYear.csv')
    
