import pandas as pd
#read data file
df = pd.read_csv('data\BigML_Dataset_5f50a4cc0d052e40e6000034.csv')

#check # of rows and columns
df.shape

#grab all column names
df.columns

#grab preview for data selection
print(df.head())