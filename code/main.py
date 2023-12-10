#pandas, numpy, and matplot imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#linear regression model imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#read data file
df = pd.read_csv('data\\BigML_Dataset_5f50a4cc0d052e40e6000034.csv')

#check # of rows and columns
#df.shape

#grab all column names
#df.columns

#grab preview for data selection
#print(df.head())

# Check for missing values
#print("Missing values:\n", df.isnull().sum())

# Drop rows with missing values (if needed)
df = df.dropna()

# Check for duplicates
#print("Duplicate rows:", df.duplicated().sum())
# Remove duplicates
df = df.drop_duplicates()

#Check data type
type(df)


# Decided to define a function to use instead of just having two very long pieces of practically the same code for question 1 and 2.
# made the code easier to read and less complicated for me to work on
def linear_regression_function(X, Y, xlabel, title, legend_loc = 'upper right'):
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=12)

    # Create linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, Y_train)

    # Make predictions
    Y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    # Plot the line of regression
    plt.scatter(X_test, Y_test, color='black', label='Real\nData\nPoints')
    plt.plot(X_test, Y_pred, color='red', label='Regress\n-ionLine')
    plt.xlabel(xlabel)
    plt.ylabel('Power generated')
    plt.title(f'Linear regression of {xlabel} vs Power generated')
    plt.legend(loc = legend_loc, fontsize='small')
    plt.show()

    # Print out the performance and coefficients
    print(xlabel,":")
    print("Model coefficients: ", model.coef_)
    print("Mean squared error: ", mse)
    print("R squared: ", r2)
    print()



# Question 1: Does sky cover affect the amount of energy generated?
print("Question 1: Does sky cover affect the amount of energy generated?\n")
linear_regression_function(df[['Sky Cover']], df[['Power Generated']], 'Sky cover', 'Linear regression of Sky cover vs Power generated')

# Answering Question 1:
print("The negative coefficient for 'Sky Cover' suggests an inverse relationship, ie. as 'Sky Cover' increases the 'Power Generated' decreases\n"
      "The low R^2 value shows that the model doesn't capture all the factors that influence 'Power Generated'\n")

# Question 2: Does wind speed affect the amount of energy generated?
print("Question 2: Does wind speed affect the amount of energy generated?\n")
linear_regression_function(df[['Average Wind Speed (Day)']], df[['Power Generated']], 'Average Wind Speed (Day)', 'Linear regression of Average Wind Speed (Day) vs Power generated', 'upper left')

# Answering Question 2:
print("The model coefficient suggests that the wind speed and power generated has a positive relationship, ie. as wind speed increases the 'Power Generated' increases too,\n"
      "As the mse is a really high value that means that there is a lot of variability in the model.\n"
      "The R^2 value suggests that the wind speed is not a significant factor in the 'Power Generated'.")

#Question 3: What is the most important weather condition for energy generation?
print("Question 3: What is the most important weather condition for energy generation?\n")
correlation_matrix = df.corr()
correlation_with_power = correlation_matrix['Power Generated'].sort_values(ascending=False)
print("Correlation with Power Generated:\n", correlation_with_power)

# Answering Question 3:
print("Through Question 3 we can see that there are 3-4 factors that have a relatively strong correlation with 'Power Generated'. Being:\n"
      "'Distance to Solar Noon'(strong negative correlation),\n" 
      "'is Daylight'(strong positive correlation),\n"
      "'Relative Humidity'(strong negative correlation),\n" 
      "'Average Wind Speed (Period)(has a weaker positive correlation)'\n")

# Additional models based on the results of Question 3
linear_regression_function(df[['Distance to Solar Noon']], df[['Power Generated']], 'Distance to Solar Noon', 'Linear regression of Distance to Solar Noon vs Power generated')
linear_regression_function(df[['Is Daylight']], df[['Power Generated']], 'Is Daylight', 'Linear regression of Is Daylight vs Power generated', 'upper left')
linear_regression_function(df[['Relative Humidity']], df[['Power Generated']], 'Relative Humidity', 'Linear regression of Relative Humidity vs Power generated')
linear_regression_function(df[['Average Wind Speed (Period)']], df[['Power Generated']], 'Average Wind Speed (Period)', 'Linear regression of Average Wind Speed (Period) vs Power generated')

