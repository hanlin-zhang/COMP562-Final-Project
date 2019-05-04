
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re


# In[2]:


import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()


# In[3]:


import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)


# In[4]:


train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')


# In[5]:


train.describe()


# In[6]:


train.head(10)


# In[7]:


def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus


# In[8]:


train['text']=clean_review(train.Phrase.values)
test['text']=clean_review(test.Phrase.values)


# In[9]:


y = train['Sentiment']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(train.text.values,y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)


# In[11]:


vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
full_text = list(train['text'].values)
vectorizer.fit(full_text)


# In[12]:


X_train_tfv =  vectorizer.transform(X_train)
X_test_tfv = vectorizer.transform(X_test)
test_tfv = vectorizer.transform(test['text'].values)


# In[13]:


lr = LogisticRegression(C=1.0)
lr.fit(X_train_tfv, y_train)
predictions1 = lr.predict(X_test_tfv)
print("accuracy_score",accuracy_score(y_test, predictions1))


# In[14]:


# pred1 = lr.predict(test_tfv)


# In[15]:


# pred_res1 = pd.DataFrame()
# pred_res1['PhraseId'] = test['PhraseId']
# pred_res1['Sentiment'] = pd.DataFrame(pred1)


# In[16]:


# pred_res1.to_csv('pred_res1.csv',index = False)


# In[17]:


svc = LinearSVC()
svc.fit(X_train_tfv, y_train)
predictions2 = svc.predict(X_test_tfv)
print("accuracy_score",accuracy_score(y_test, predictions2))


# In[18]:


pred2 = svc.predict(test_tfv)


# In[19]:


pred_res2 = pd.DataFrame()
pred_res2['PhraseId'] = test['PhraseId']
pred_res2['Sentiment'] = pd.DataFrame(pred2)


# In[20]:


pred_res2.to_csv('pred_res2.csv',index = False)

