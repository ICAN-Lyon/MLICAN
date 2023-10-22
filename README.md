# MLICAN
Introduction to Machine Learning IA Design Week ICAN


### TODO Create your first model
To create a notebook in python use the extension .ipynb
Nothing is require in Google Colab

first step : assign a variable named data that will take the function as value
pd.read_csv("celsius.csv")

second step : use the function .info() on data to check the metadata of the celsius.csv

third step : use the function .head() on data to have an overview of the first rows of the table

fourth step : import seaborn as sb

fifth step : call the fonction scatterplot() on sb to draw a graphic of the relation between fahrenheit and celsius values. *scatterplot function take multiples parameters.
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

from sklearn.linear_model import LinearRegression
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

OK, fine now practices, collects enough data to have 3 csv with one caracteristic and a label on each one, then train a model to predict labels.
Then print the score of your model. 
Example do it for a relation between height and weight. Find to more dataset  models. 
Then publish it on your repository with the "databases"
