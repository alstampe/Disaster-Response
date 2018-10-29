import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
import plotly.graph_objs as go
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Data', engine)
conn = engine.connect()
#df = pd.read_sql("SELECT * FROM df", con=conn)

#engine = create_engine('sqlite:///.../data/DisasterResponse.db')
#df = pd.read_sql("SELECT * FROM df", engine)
#X = df['message']
#Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)



#load model
#model = joblib.load("../models/mymodel.pkl")
#model = joblib.load("../models/model_filepath")
#model = joblib.load("../models/Mymodel")
model = joblib.load("../models/Model")

print(model)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #ex
    #exend
    
    # create visuals
    #Testing
     # Show distribution of different category
    '''
    category = list(df.columns[4:])
    category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))

    # extract data exclude related
    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:11]
    categories_names = list(categories_mean.index)
    #END testing '''
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    #testplot
   
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    print('QUery=', query)

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
    print('go is called') 

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()