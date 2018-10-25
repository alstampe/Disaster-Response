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


def load_data(messages_filepath, categories_filepath):   
    messages =  pd.read_csv(messages_filepath)   
    categories = pd.read_csv(categories_filepath)   
    namestring=categories.categories[1]
    df = pd.merge(messages, categories, how='outer')
    return messages, categories, df
    pass


def clean_data(df, categories):
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
    df = pd.merge(df, expanded, right_index=True, left_on='id')
    df = df.drop(['categories'], axis=1)
    df.drop_duplicates(inplace=True)
    return df
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///Database.db')
    df.to_sql(database_filename, engine, if_exists='replace', index=False)
    
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
       # messages_filepath = disaster.messages.csv
       # categories_filepath = disaster_categories.csv
       # database_filepath = 'Data4' 

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
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()