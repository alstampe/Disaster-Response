    
 
#    IMPORT Statements for process_data.py
 
import sys
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
import sqlite3

#    Functions for process_data:
#    load_data
#    clean_data
#    save_data

def load_data(messages_filepath, categories_filepath):   
    '''
    Function will read 2 cvs files and return both separate and combined dataframes  
    Input : csv files; messages.csv and categories.csv 
    Output : dataframes 'messages' and 'categories', and the combined dataframed 'df'
    'namestring' gives catefory names, but are not returned
    '''
    messages =  pd.read_csv(messages_filepath)   
    categories = pd.read_csv(categories_filepath)   
    df = pd.merge(messages, categories, how='outer')
    return messages, categories, df
    pass

def clean_data(df, categories):
    '''
    Function cleans data and returns a cleaned dataframe
    Intermediate dataframe for cleaning : 'combined'
    'names' is the list over categorical column names
    Input: df and categories
    Output : df 
    '''
    expanded = categories['categories'].str.split(';' , expand=True)
    names=categories.categories[1]
    names=names.replace("-", "")
    names=names.replace("0", "")
    names=names.replace("1", "")
    names=names.replace(";", ",")
    names=names.replace(";", ",")
    names=names.split(',')
    expanded.columns = names
    expanded.replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
    category = list(expanded.columns)
    for column_name in category:
        expanded[[column_name]] = expanded[[column_name]].apply(pd.to_numeric)
    expanded.drop('child_alone', axis = 1, inplace = True)    
    df = pd.merge(df, expanded, right_index=True, left_on='id')
    df = df[df['related'] != 2]
    df = df.drop(['categories'], axis=1)
    df.drop_duplicates(inplace=True)
    return df



def save_data(df, database_filename):
    '''
    Function saves the cleaned dataframe in the sqlite database 'DisasterResponse.db'
    Input: df and database_filename
    Output : none
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponseData', engine, if_exists='replace', index=False)
    
    #engine = create_engine('sqlite:///DisasterResponse.db')
    #df.to_sql(database_filename, engine, if_exists='replace', index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
       
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories, df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
                
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filename in the database DisasterREsponse.db to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()