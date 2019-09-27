# MinnesotaOpioid
### Analysis of opioid prescription rates in Minnesota

## Because of the experimental nature of this analysis, I haven't written great code. It's mostly proof of concept for the tools I'm demonstrating.

Start with script StandardizeOpioid.py and, from files OpioidData????ByCounty.txt in RawDataIn/, create OpioidMNDataByCounty.csv.

NOTE that StandardizeOpioid.py right now does not understand the file structure, it is looking for the .txt (tab separated) files in the run directory, but they're actually in RawDataIn/. 

After You obtain OpioidMNDataByCounty.csv, run DataProcess.py. 

Currently DataProcess.py cleans up the data a little more, pivots the table to have a table with years as columns and counties as indices (instead of counties as indices but repeated for each year). After the pivot is completed, a matrix/tensor is created from this and that matrix is factored into yearVec and countyVec using PyTorch's autograd gradient descent algorithm. 

Lastly, cosine similarity is calculated for the set of years and also the set of counties, using yearVec and countyVec, respectively. 
