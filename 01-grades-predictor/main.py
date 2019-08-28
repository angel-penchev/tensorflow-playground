import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import pickle
from selectmenu import SelectMenu


# Reading data from .csv file
data = pd.read_csv("./data/student-mat.csv", sep=";");


# Encoding all the non-numeric data
le = preprocessing.LabelEncoder()
for column in data:
    if data[column].dtype != "int64":
        data[column] = le.fit_transform(list(data[column]));


# Setting the parameters and the lable
predict = "G3";
features = np.array(data.drop([predict], 1));
labels = np.array(data[predict]);


# Separating training data from test data
test_train_data_ratio = 1 / 10;
features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, test_size = test_train_data_ratio);


# Training the linear model
def train_model(iterations, filepath):
        best_accuracy = 0;

        for _ in range(iterations):
                linear = linear_model.LinearRegression();
                linear.fit(features_train, labels_train);
                accuracy = linear.score(features_test, labels_test);
                if accuracy > best_accuracy:
                        best_accuracy = accuracy;
                        with open(filepath, "wb") as file:
                                pickle.dump(linear, file);

        print(best_accuracy);
        return linear;


# Loading an already trained model
def load_model(filepath):
        with open(filepath, "rb") as file:
                return pickle.load(file);


# Displaying a function selection menu
menu = SelectMenu();
menu.add_choices(["Train a new model", "Load an exsisting model"]);
selected = menu.select("What do you want to do?");

if (selected == "Train a new model"): linear = train_model(100, "./model/grades-predictor.pickle");
elif (selected == "What do you want to do?"): linear = load_model("./model/grades-predictor.pickle");


# Getting a list of all the predictions and their real labels
predictions = linear.predict(features_test) 
for x in range(len(predictions)):
    print("% .5f - %d" % (predictions[x], labels_test[x]))