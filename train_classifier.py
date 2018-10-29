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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
import pickle
import sqlite3
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

def load_data():
    engine = create_engine('qlite:///../data/DisasterResponse.db')
    conn = engine.connect()
    df = pd.read_sql("SELECT * FROM Data", con=conn)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(df.columns)
    return df, X, y, category_names

#def load_data(database_filepath):
#    #engine = create_engine('sqlite:///DisasterResponse.db')
#    engine = create_engine('sqlite:///../data/DisasterResponse.db')
#    conn = engine.connect()
#    df = pd.read_sql("SELECT * FROM Data", con=conn)
#    X = df.message.values
#    y = df[df.columns[4:]].values
#    df = df.drop(['id', 'message', 'original', 'genre', 'related', 'child_alone'], axis=1)
    #category_names = list(df.columns[4:])
#    category_names = list(df.columns)
    
#    return X, y, category_names
#    pass


def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    pass


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 25], 
              'clf__estimator__min_samples_split':[2, 5, 10]}
    
  #  parameters = {
  #      'vect__ngram_range': ((1, 1), (1, 2)),
  #      'clf__estimator__min_samples_split': [2, 4],
  #      'vect__max_df': (0.5, 0.75, 1.0)
  #  }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2)
    return cv



def get_eval_metrics(actual, predicted, col_names):
    """Calculate evaluation metrics for ML model
    
    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: list of strings. List containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df


def evaluate_model(model, X_test, y_test, category_names):

    """Returns test accuracy, precision, recall and F1 score for fitted model 
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.  
    Returns:
    None
    """
    # Predict labels for test dataset
    y_pred = model.predict(X_test)   
    # Calculate and print evaluation metrics
    eval_metrics = get_eval_metrics(np.array(y_test), y_pred, category_names)
    print(eval_metrics)

    
def evaluate_model_2(model, X_test, y_test, category_names):
    """
    Evaluates the trained model by predicting on test data
    The classification report prints output accuracy, precision and recall for all categories
    """
    y_pred = model.predict(X_test)
    y_test_df= pd.DataFrame(data=y_test)
    y_pred_df= pd.DataFrame(data=y_pred)    
    for i in range(0, len(category_names)):
        print("Category:", category_names[i])
        print(classification_report(y_test_df.values[:,i], y_pred_df.values[:,i]))
    pass


def save_model(model, model_filepath):
    # open the file for writing
     # this writes the object a to the file named 'model'
     # here we close the fileObject    
    #file_Name = 'model_filepath'   
    #fileObject = open(file_Name,'wb')    
    #pickle.dump(model,fileObject)    
    #fileObject.close()
    pickle.dump(model, open(model_filepath, "wb")) 
    print('pickle model saved in:', model_filepath)
    joblib.dump(model, 'classifier.pkl')
    print('joblib model also saved , as classifier, saved in:', 'classifier.pkl')
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)
        print('Evaluating model again...')
        evaluate_model_2(model, X_test, y_test, category_names)
        
        print('Model evaluated')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()