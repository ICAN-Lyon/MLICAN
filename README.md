# MLICAN
Introduction to Machine Learning IA Design Week ICAN


###  EX 01 TODO Create your first model
To create a notebook in python use the extension .ipynb
Nothing is require in Google Colab

first step : assign a variable named data that will take the function as value
pd.read_csv("celsius.csv")

second step : use the function .info() on data to check the metadata of the celsius.csv

third step : use the function .head() on data to have an overview of the first rows of the table

fourth step : import seaborn as sb

fifth step : use the fonction scatterplot() on data to draw a graphic of the relation between fahrenheit and celsius values. *scatterplot function take multiples parameters.
first parameter should have this pattern [x="celsius], second one [y="fahrenheit"], third one data=data the second data is the name of the function which reads the csv.file

Tips use hue="fahrenheit" as fourth parameter to colorize points depending on the fahrenheit values. 

Tips you can also change the color palette like this as a fifth parameter palette="coolwarm"

step 6 : You need to separate the caracteristics and the labels. 
caracteristics are on the x axis the celsius value, and your labels are 
on the y axis the fahrenheit values. 

X = data["celsius"]
Y = data["fahrenheit"]

run the X on the notebook, you should be able to see all the values with each index
Same for the Y.

The array type doesnt fit the required data structure we need. 
Check the type of x with the function : type(X)
As you can see we need some adjustment. 
Instead of having an array with all the values insides, we need an array of an array that contains each one on their position. 
Run X.values should be like this [-40, -10, 0, 8, ...]
We need this format [[-40],[-10],[0],[...]]

Use numpy function reshape
X.values.reshape(-1,1)

Save the result into a new variable x_processed do the same with Y.values
into y_processed

step 7 : we are going to create our model and for that we are going to use from sklearn.linear

import sklearn.linear_model import LinearRegression
assign the function LinearRegression into a model variable
model = LinearRegression()
Be aware that our model cant be use yet, because he wasnt trained yet !
In order to train it he needs to receive the x values and the y values.
But there is a function to train it.

Have a look : 
model.fit(x_processed, y_processed)
Run it on the notebook.

Now we can use the model to predict fahrenheit values outside the table.
Here is the example : 

model.predict([[8]])
Here we try it, with a known value to see if our model prediction is right.
It should predict 46.4 fahrenheit.

Try it now with a different values.
The fact that our graphic is a linear function, it have a perfect relation, the score will be 1 which is not common for real life works models. And the values predicted will always be correct regarding the converting formula. Because this is a simple model with low relationship and patterns. 

You can check the model score like this  :
model.score(x_processed,y_processed)
This will show a number between 0 and 1.

###  EX 02 CALIFORNIA HOUSING PRICES
[this is an external link to kaggle.com][https://www.kaggle.com/datasets/camnugent/california-housing-prices]

##TODO 

import the csv and read it

import pandas as pd
then
data = pd.read_csv("housing.csv")
use the function head on data

print data["ocean_proximity"] 
Houston we have a problem !
Let's fix it 
data["ocean_proximity"].value_counts()

Then you will see all the diferent data with their number count.
However, machine learning models cannot simply take this data and process it ...
Therefore, later we will have to transform this column into numeric data.

Well, have a look at our data using data.info()
You will see how much non-empty data comes and the type of data
You will see that total_bedrooms have 200 empty data.
Either we fill the 200 missing or drop it.

use data.decribe()
To have more information about our data.
count is the number of records,
mean the average
The percent value are the sorted ascending values. 
For example at 25% the median_income is  25 000. 

A graph is more explanatory than records
draw a graph with hist()
data.hist()
Fix the result using data.hist(figsize=(15,8), bins=30, edgecolor="black")
Check the documentation of seaborn and pandas if you want more parameters.

Keep going
import seaborn as sb
then draw a scatterplot with the latitude and longitude
sb.scatterplot(x="latitude",y="longitude", data=data)

Then you will see a cloudpoints. At first glance, you may not recognize
something. But remember theses are latitude points and longitude points.
So it is a map of California ! 

Make the hue parameter taking the median_house_value
hue="median_house_value"
palette="coolwarm" 
Then analyse again or data, what do you see.

use another paramater s=data["population"]
Oops we broke something let's fix it. 
s=data["population"]/100

Lets have a look on how we can filter this data, as Stephen Hawking said, data should be on the right side, let's go on the darkside.

Imagine we want to rob the wealthy people of this area, how can we see the data ?
try this ==> data.data[(data.median_income > 14)]

Here you have the people earning more than 140K a year 
Nice nice .....

Back to serious thing, remember the empty records
use the function data.dropna() to drop the empty records
if you want to overwrite the data
use this : data_na = data.dropna()

Something it is better not to overwrite the original data.
use it again : data_na.info()

There should be no empty records

Lets convert the numerical feature regarding the ocean distance. 
data_na["ocean_proximity"]
data_na["ocean_proximity"].value.counts()

Assign numerical values assuming the line of records
1H OCEAN should be 1
INLAND should be 2
NEAR OCEAN 3
NEAR BAY 4
ISLAND 5

But 1,2,3,4,5 are numerical values, and the model could interpret that 1 is lower than 5
So i'll introduce you to dummies.

Dummies or One-hot encoding are represented as the following to convert the data
NEAR BAY    INLAND  NEAR OCEAN
    1           0       0
    0           0       1

Using dummies we do not have numerical order, but only 1 or 0 values.
Read pandas documentation to learn more about dummies.

use pd.get_dummies(data_na["ocean_proximity"])
As you can see it fixes the problem.

Sometimes dummies can return a boolean if this is the case give the following parameter.
pd.get_dummies(data_na["ocean_proximity"], dtype=int)

assign the result into a dummies variable
dummies = pd.get_dummies(data_na["ocean_proximity"], dtype=int)

let's join ou data
data_na.join(dummies)
This function returns a new dataset as you can see
data_na = data_na.join(dummies)

Confirm the join doing data_na.(head) but wait ocean_proximity is fixed but we still have the original column.

use the drop()
data_na.drop(["ocean_proximity"], axis=1)
axis=1 is use to delete column

data_na = data_na.drop(["ocean_proximity"], axis=1)

There is still a plenty more analysis we can do, but we need the correlations.
Use the function data.corr() ok nice table try to understand it.

Now we should use seaborn to draw a heatmap
sb.heatmap(data_na.corr())

Nice, but it will be more readable if we could have number inside the heatmap.
sb.heatmap(data_na.corr(), annot=True)
Okay lets fix it

sb.set(rc={"figure.figsize : (15,8)})
sb.heatmap(data_na.corr(), annot=True, cmap="YlGnBu")

if you want to focus on a specific label, use this formula to see the correlation between the others labels
data.corr()["median_house_value].sort_values(ascending=False)

Important notice, this is not magic number, but a reading of the documentation.
Reading manuals, documentation is part of the software engineer path. It is OVERPOWERED OVER 9000 !!!!

###  EX 03 Titanic
