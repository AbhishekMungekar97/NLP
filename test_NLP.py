# import required packages
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.datasets import imdb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from bs4 import BeautifulSoup as html_removal
import re,string,unicodedata

# loading the IMDB dataset  
dataset =pd.read_csv('data/aclImdb_v1.csv')

#Assigning integer based values to the postive and negative sentiments
labels = preprocessing.LabelEncoder()
dataset['sentiment'] = labels.fit_transform(dataset['sentiment'])

#Data Pre-processing
print('======================================================================================')
print('The Reviews are being Pre-processed..........')
print('======================================================================================')

# Removing any form of UHTML strips from the reviews
def HTML_removal(Review):
    clean_data = html_removal(Review, "html.parser")
    return clean_data.get_text()

#Removing all the breackets from the reviews
def Parentheses_removal(Review):
    return re.sub('\[[^]]*\]', '', Review)

# Removing all the special characters from the Reviews
def special_character_removal(Review, remove_digits=True):
    special_characters=r'[^a-zA-z0-9\s]'
    return re.sub(special_characters,'',Review)

#Applying all the above pre-processing measures
def Review_processing(Review):
    Review = HTML_removal(Review)
    Review = Parentheses_removal(Review)
    Review = special_character_removal(Review)
    return Review

dataset['review']=dataset['review'].apply(Review_processing)
 
# Vectorizing the top 35k words from the IMDB reviews dataset, and converting the sentiment indicators into a float
print('======================================================================================')
print('The Testing Reviews are being vectorized..........')
print('======================================================================================')
def Embedding(Review, Sentiment):
    kwargs = {'ngram_range' : (1, 2),'dtype' : 'int32','strip_accents' : 'unicode','decode_error' : 'replace','analyzer' : 'word','min_df' : 2}
    Vectorizer = TfidfVectorizer(**kwargs)
    Review_vectorized = Vectorizer.fit_transform(Review)
    selector = SelectKBest(f_classif, k=min(35000, Review_vectorized.shape[1]))
    selector.fit(Review_vectorized,Sentiment)
    Review_vectorized = selector.transform(Review_vectorized).astype('float32')
    return Review_vectorized

Embedded_data = Embedding(dataset['review'], dataset['sentiment'])
# Splitting the dataset into 50% training and 50% testing
X = Embedded_data.toarray()
y = (np.array(dataset['sentiment']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

if __name__ == "__main__": 
    
    #Load the model of choice
    print('======================================================================================')
    print('The Saved model is being uploaded......................')
    print('======================================================================================')
    model = keras.models.load_model('models/20867324_NLP_model.h5')
    
    # Run prediction on the test data and output required plot and loss
    predicted_value= model.predict(X_test)
    predictions =[]
    for i in predicted_value :
        if i >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
        
    # The accuracy of the test dataset is as follows :-
    print('======================================================================================')
    k = accuracy_score(y_test, predictions)
    print('The accuracy of the predicted sentiments of the IMDB review set is :-',k)
    print('======================================================================================')
    
