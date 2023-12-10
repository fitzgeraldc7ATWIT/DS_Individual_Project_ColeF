# Solar Data Science Project

## Introduction

The objective of this project is to use measurements to predict the output of a solar power system installed in Berkely, CA. I was always curious about how different conditions can affect the output of solar powered systems.

I was inspired to do this project simply because I find solar power and energy to be quite interesting. I would really like to in the future see a more efficient solar system be created because of the potential I feel that solar power has.

## Data Selection

The data has over 2,900 samples with 16 columns: 
'Day of Year','Year', 'Month', 'Day', 'First Hour of Period','Is Daylight', 'Distance to Solar Noon', 'Average Temperature (Day)','Average Wind Direction (Day)', 'Average Wind Speed (Day)', 'Sky Cover',   'Visibility', 'Relative Humidity', 'Average Wind Speed (Period)','Average Barometric Pressure (Period)', 'Power Generated'.

The purpose being a model created to predict the output of a solar power system. The dataset can be found online at [kaggle](https://www.kaggle.com/datasets/vipulgote4/solar-power-generation/)

Data Preview:
![Data screenshot](./graph/dataPreview.png)

## Methods

Tools:
- NumPy, Pandas, Matplotlib, Scikit-learn
- GitHub
- VS Code as IDE

Methods used with Scikit-learn:
- Linear regressin model

"Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable." - [IBM](https://www.ibm.com/topics/linear-regression). 

I chose the linear regression model because I wanted to explore the relationship between weather conditions and the power generation of this solar power system based in California. 

I assumed with this method I would be able to find out the most important factors in generating the most power and being able to show that with a graph of the line of regression. 

I used the Scikit-learns linear regression model in this project. I used the LinearRegression class for model training and used functions like fit() and predict() to process the data.

