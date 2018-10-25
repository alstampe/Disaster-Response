# Disaster-Response
#Project for creating a pipeline 
Project outline, as given in project description

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

Completed steps 1-7 in ETL Pipeline preparation in Notebook
Dataframe loaded into sql Database 

New status:
Completed most steps in Part 2; building the pipeline
The data was read from SQLlite, and I stored the final model in a Pickle file.
Have some challenges sorting out the correct labels when predicting
Will start testing on different parameters 
