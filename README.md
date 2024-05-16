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
sb.scatterplot(x="celsius",y="fahrenheit",data=data, hue="fahrenheit",palette="coolwarm")

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
https://www.kaggle.com/c/titanic/data?select=train.csv

Let's start with the same routine, we had previously. 
import pandas, set the alias, and read the file, and show the first items.

import pandas as pd
data = pd.read_csv("train.csv")
data.head()

Well, lets analyse our dataset, we can say, that many column wont be relevant to our dataset, and it could possibly go wrong using these kind of data. 
Name, won't be relevant, same as cabin and Ticket and passengerID. Because they might be contained in others data, like gender, or fare or class. Ill give you an quick explanation IRL. 
Check the categories in Kaggle if you dont understand a column. 

use data.describe()
import seaborn and try to figure out how many survived. 

import seaborn as sb
sb.countplot(x="Survived")
Now you can see two columns, try to show another graphic representation, with genders.

sb.countplot(x='Survived',data =data, hue="Sex")

Now you have the graphic showing how many males and females survived and died in the Titanic.

Use .isna() to see if we have void values. 
then use .sum() to see a better result. 

data.isna().sum()
We have a little problem, as we can see cabin is irrelevant, and a lot of data is missing, age too, and embarked a fewer. We might use a mean in age values. 

Let's try to show a graphic with the ages of passengers. 
sb.displot(x="Age",data=data)

Lets try to fir our record of 177 void data, with the median age of passengers. 
data["Age"]

this will show a list of records.
As we can see some are Nan, 

Lets calculate the mean
data["Age"].mean()

Remember when we used dropna function, well there is one function to fill too.
so we want to fill our void records with the mean age calculated below. 

data["Age"].fillna(data["Age"].mean())
lets assign this into our new variable
data["Age"] = data["Age"].fillna(data["Age"].mean())

Lets check if this worked, 

data.isna().sum()
All fine. 

Lets drop the cabin. 
data = data.drop(["Cabin"], axis=1)

let see our other values, that we wanted to change, 

data["Embarked"].value_counts()

time to clean 

data = data.dropna()
data.isna().sum()

Perfect. But
Lets drop the name, passengerd and tickets

data = data.drop(["Name", "PassengerId", "Ticket"], axis =1)

confirm the drop, by using the command, 

data.head()
You wont see our tree beloved columns. 
But remember ML algorithm always try to make relation and give weight to numeric data like Numbers. the more data we have as number, the better the relation would be and be give by the algorithm. 
We can convert Sex gender to numeric values. 0 for Woman, 1 for Male. Yeah Data can be sometimes gendered racist. 

Lets use our old friends, the dummies 
pd.get_dummies(data["Sex"])

Lets drop the first column, we only need one. 

pd.get_dummies(data["Sex"], drop_first=True)
Why do we drop the first column ? In fact we want to avoid a more complex problem that we call "Multicollinearity", this would had made troubles in our dataset and predictions, because, the algorithm would had try to make a relation between male and female and their values. 
And we do not want to. 

Lets our filtered column into a new variable. 

dummies_sex = pd.get_dummies(data["Sex"], drop_first=True)

then join it to our data 

data = data.join(dummies_sex)
drop the old column 
data = data.drop(["Sex"], axis=1)

Lets try to clean the Embarked colums the same way we did for Sex. 

dummies_embarked = pd.get_dummies(data["Embarked"], drop_first=True)
data = data.join(dummies_embarked)
data = data.drop(["Embarked"],axis=1)

Let's see the correlation between our data.

sb.heatmap(data.corr(),annot=True,cmap="YlGnBu")
Try to understand, if you dont and need another graphic try our countplot

sb.countplot(x="Survived", data=data, hue="Pclass")
Lets try to train our data 

X = data.drop(["Survived"], axis=1)
y = data["Survived"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

Lets try some prediction

prediction = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)

The result here could be problematic, because our dataset is not perfect and unbalanced, the reason is that we do not have the same number of passengers that died and survived. Many died, and few survived
We can adjust this accuracy by adding some classification. 

from sklearn.metrics import classification_report

print(classification_report(y_test,prediction))
check the documentation if you need a better understanding. 

Let import a confusion matrix, this would compare when our prediction failed compared to reality

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, prediction)
this give an array use panda 

pd.DataFrame(confusion_matrix(y_test, prediction),columns=["Pred: No", "Pred: Yes"], index=["Real: No", "Real: Yes"])
pd.DataFrame(confusion_matrix(y_test, prediction),columns=["Pred: No", "Pred: Yes"], index=["Real: No", "Real: Yes"])

X.head()
new_passenger = [3,50,0,0,100,False,False,False]



prediction = model.predict([new_passenger])
if prediction[0] ==1 :
  print("You survived")
else : 
  print("You died")

