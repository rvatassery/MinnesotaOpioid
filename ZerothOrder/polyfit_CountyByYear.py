#!/usr/bin/python3

import numpy, pandas, os, sys, scipy

def polyfitDF(DF,xcol,ycol,deg):
    polyfitArray = numpy.polyfit(DF[xcol],DF[ycol],deg,full=True)
    return polyfitArray

if __name__ == '__main__':
    
    # sys.argv[1] should be 'OpioidMNDataByCounty.csv'
    newDF = pandas.read_csv(sys.argv[1])

    # create the average table across all years, performed in iPython CLI
    '''
    countyAvg = newDF.groupby(['County'])['TotalPrescripts'].mean()
    countyAvg.index = countyAvg.index.str.replace(' ','_')
    countyAvg.sort_values(inplace=True)
    countyAvg.to_csv('CountyAverageAllYears.csv',header=True,sep=' ')
    '''
    polyResult = newDF.groupby(['County']).apply(polyfitDF,
                                xcol='year',ycol='TotalPrescripts',deg=2)
    polyResultDF = polyResult.reset_index()
    polyResultDF.columns = ['County','polyFitResult']

    holderDF = polyResultDF.polyFitResult.apply(pandas.Series)
    holderDF.columns = ['polyfit','resid','rank','singVals','rcond']
    holderDF1 = holderDF.polyfit.apply(pandas.Series)
    holderDF1.columns = ['a','b','c']
    polyResultDF = pandas.concat([polyResultDF,holderDF1],axis=1)
    polyResultDF.dropna(subset=['a'],axis=0,inplace=True)
    polyResultDF.sort_values(by=['a'],inplace=True)
    polyResultDF.to_csv('polyFitByCounty.csv',index=False)
