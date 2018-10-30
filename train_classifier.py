# IMPORT statements for Train_Classifier

import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
import pickle
import sqlite3
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Functions; load_data, tokenize, build_model, get_eval_metrics, evaluate_model, evaluate_model2, save_model 

def load_data(database_file):
    '''
    Function loads data from database file
    Input : database_file (filename)
    df is kept in its original in 'df' 
    X and y are extracted as subsets; 
    X contains messages and y contains values from category columns only   
    Output : df, X, y and category_names
    
    '''
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    conn = engine.connect()
    df = pd.read_sql("SELECT * FROM Data", con=conn)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(y.columns)
    return df, X, y, category_names

def tokenize(text):
    '''
    Function performs tokenization on text strings
    Input : text (text strings from messages)
    Output : processed text    (tokenized and lemmatized)
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    pass


def build_model():
    '''
    Function builds model and pipeline , using GridSearch and a parameter set for testing
    RandomForestClassifier is used for classification
    Input : None
    Output : cv (model, a GridSearch produced object)
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 25], 
              'clf__estimator__min_samples_split':[2, 5, 10]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2)
    return cv



def get_eval_metrics(actual, predicted, col_names):
    """
    Function calculates an evaluation metrics for the model
    
    Input:    actual: array. Array containing actual label.
              predicted: array. Array containing predicted labels
              col_names: list of strings, names names for each of the predicted fields, called categories
       
    Output:   metrics_df: dataframe containing the accuracy, precision, recall 
              and f1 score for a given set of actual and predicted labels
              
    This evaluation metrics setup is written based on a GitHub project by Genevieve Hayes/gkhayes
    The use of the metrics setup (below) is also based on her project.
    """
    metrics = []
    
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df


def evaluate_model(model, X_test, y_test, category_names):

    """
    Calculates test accuracy, precision, recall and F1 score for the fitted model 
   Input:   model: fitted model object from buil_model and train_model
            X_test: dataframe, containing test dataset
            y_test: dataframe, containing the test labels 
    category_names: list containing all category names.  
    Prints results in program
    Returns: None
    """
    y_pred = model.predict(X_test)   
    eval_metrics = get_eval_metrics(np.array(y_test), y_pred, category_names)
    print(eval_metrics)

    
def evaluate_model_2(model, X_test, y_test, category_names):
    """
    Evaluates the trained model by predicting on test data
    The classification report prints output accuracy, precision and recall for all categories
    This function delivers the same as the above, but in different output formats.
    I struggled to debug this function and therefore wrote the aabove based on a GitHub project,
    but finally managed to get this right and decided to keep both. 
    """
    y_pred = model.predict(X_test)
    y_test_df= pd.DataFrame(data=y_test)
    y_pred_df= pd.DataFrame(data=y_pred)    
    for i in range(0, len(category_names)):
        print("Category:", category_names[i])
        print(classification_report(y_test_df.values[:,i], y_pred_df.values[:,i]))
    pass


def save_model(model, model_filepath):
    '''
    Function will save the model for use in the app
    Note that it contains two save codes; one using pickle and one using joblib
    I kept both after testing to have a backup, as the train/fit process takes a very long time to rerun ..
    The joblib version will always be saved as classifier.pkl , I usually use 'Classifier.pkl for the pickle version)
    Input: model, model_filepart (filename)
    Output:None
    '''
    pickle.dump(model, open(model_filepath, "wb")) 
    print('pickle model saved in:', model_filepath)
    joblib.dump(model, 'classifier.pkl')
    print('joblib model also saved , as classifier, saved in:', 'classifier.pkl')
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df, X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved first time!')
        
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)
        print('Evaluating model again...')
        evaluate_model_2(model, X_test, y_test, category_names)
        
        print('Model evaluated')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved again!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
