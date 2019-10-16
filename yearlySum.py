#!/usr/bin/python3

import os, sys, re, inspect, pandas, glob, numpy

mnDB = pandas.read_csv('OpioidMNDataByCountyPivot.csv',sep='\t')
colList = list(mnDB.columns)
print(colList)
colList.remove('County')
print(mnDB[colList])
mnDB_agg = mnDB[colList].sum()
mnDB_agg.to_csv('OpioidMNSumYear.csv',sep='\t',index=False)
