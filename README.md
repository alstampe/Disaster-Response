# Disaster-Response app 
Project using ML for predicting correct category disaster-related messages 
Project is a part of a Udacity Nanodegree

# Framework for project
Project is performed using Udacity provided JupyterNotebook and workspace IDE
Data input is provided by .csv files and we use prepared code structores with some precoded HTML 

# Project outline 
Project outline, as given in project description

- app
 - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model 

- README.md

# App description
The purpose of the porject is to code and train a classification model, based on Machine Learning algorithms.
The app shall categorize a free-format sms text and choose the most relevant categpry among 36 predefine categories.
The texts will also be sorted in 3 genres. 
Visuals is provided in the app to show information on the messages contained in the dataset.

# Files
python code : process_data.py, train_classifier.py, run.py

data : disaster_categories.csv, disaster_messages.csv, DisasterDatabase.db/Data and Classifier.pkl

app : as provided

the files are both in root directory and sorted in folders as they were in the Udacity workspace
  

# How to run the program
1. Prepare data : Run : python process_data.py disaster_categories.csv disaster_messages.csv DisasterResponse.db
   The code will expect the csv files to be in the same directory 
2) Build, train and save classifier (expected to be in the same directory): python train_classifier.py Data Classifier
   The load function specifies the path to the data folder from the workspace, can easily be altered 
3) Use the app for classifying texts (expected to be in an app folder): python run.py
   Open the app in a browser : https://view6914b2f4-3001.udacity-student-workspaces.com/
   In addition to the one visualisation included in the given material, I have added 4 more figures
   Test by entering random text strings and confirm classification by revirewing the result in the app

# Comments
In spite of repeated trainings with and without GridSearch and quite good accuracy on individual categories, the app does not perform well in predicting the best category. This is probably a result of a heavily skewed dataset, dominated by 'related' - and the fact that, in general, we have few occurences of 'hits' (positive values) for the majority of categories, making the classifier weak in recognizing the right category even when the text should be very clearly identifying cases like 'fire', 'storm', floods' etc.    
