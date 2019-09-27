#!/usr/bin/python3

import os, sys, re, inspect, pandas, glob, numpy

opioidDict = {}
for a in range(2006,2018):
    print('Processing year ',a)
    opioidDict[a] = pandas.read_csv('OpioidData'+str(a)+'ByCounty.txt',
                                    sep='\t',encoding='ISO-8859-1',
                                    dtype={'County':str,'State':str})
    opioidDict[a].columns = ['County','State','FIPSCounty','TotalPrescripts']
    if opioidDict[a].TotalPrescripts.dtype != 'float64':
        opioidDict[a].TotalPrescripts = \
                            opioidDict[a].TotalPrescripts.str.replace('Ð','NA')
    opioidDict[a]['year'] = a
    opioidDict[a]['County'] = numpy.where(opioidDict[a].County.str.contains(',')
                        ,opioidDict[a].County.str.split(',').str[0].str.upper(),
                        opioidDict[a].County.str.upper())

opioidDF = pandas.DataFrame()
for DF in opioidDict:
    opioidDF = pandas.concat([opioidDF,opioidDict[DF]],ignore_index=True)

opioidDF.to_csv('OpioidDataByCounty.csv',index=False)
opioidDF[opioidDF.State=='MN'].to_csv('OpioidMNDataByCounty.csv',index=False)
