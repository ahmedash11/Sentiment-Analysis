#!/usr/bin/env python
# coding: utf-8

# In[3]:


import csv
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from string import punctuation
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from difflib import SequenceMatcher


# # Start of Classifying Using PreProcessing only

# In[4]:


tweets = pd.read_csv('./Tweets.csv', encoding='utf-8')


# In[5]:


df_ = pd.DataFrame(tweets, columns=['text','airline_sentiment'])


# In[6]:


texts = df_['text']
sentiment = df_['airline_sentiment']


# In[7]:


# remove urls and strip
texts = [re.sub(r'https?:\/\/.*[\r\n]*', '', text).strip() for text in texts]
len(texts)


# In[8]:


# remove unordinal characters
texts = [''.join(character for character in text if ord(character) < 128) for text in texts]
len(texts)


# In[9]:


# remove punctuations
punctuation += "“’…”"
texts = [''.join(character for character in text.decode('utf-8').encode('ascii') if character not in punctuation) for text in texts]
len(texts)


# In[10]:


# tokenizing every single text
tokenized = []
for text in texts:
    tokenized.append(word_tokenize(text))


# In[11]:


# Stemming every token 
snowball_stemmer = nltk.stem.SnowballStemmer('english')
tokenized = [[snowball_stemmer.stem(word) for word in words] for words in tokenized]


# In[12]:


# removing every stopword 
tokenized = [[word for word in words if word not in stopwords.words('english')] for words in tokenized]


# In[13]:


# appending words together forming a sentence 
finalTexts = [' '.join(words) for words in tokenized]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(finalTexts, sentiment, test_size=0.2, random_state=1)


# ## Applying TFIDF Vectorizer

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

# Calculating the tf-idf scores
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)


# ## NaiveBayes without Filtering

# In[16]:


from sklearn.naive_bayes import MultinomialNB

# Initialize classifier
naive_bayes_clf = MultinomialNB()

# Train classifier
naive_bayes_clf.fit(X_train_transformed, y_train)


# In[17]:


print(naive_bayes_clf.score(X_test_transformed, y_test))


# ## F1 Score NaiveBayes Without Filtering

# In[18]:


# F1 Score NaiveBayes
from sklearn.metrics import f1_score

y_pred = naive_bayes_clf.predict(X_test_transformed)

score = f1_score(y_test, y_pred, average='micro')
print(score)


# ## KNeighbors Classifier without Filtering

# In[19]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize classifier
k_nearest_clf = KNeighborsClassifier(n_neighbors=3)

# Train classifier
k_nearest_clf.fit(X_train_transformed, y_train)


# ### F1 Score KNeighbors Classifier without Filtering

# In[20]:


from sklearn.metrics import f1_score

y_pred = k_nearest_clf.predict(X_test_transformed)

score = f1_score(y_test, y_pred, average='micro')
print(score)


# ## Random Forest Classifier without Filtering

# In[21]:


from sklearn.ensemble import RandomForestClassifier

# Initialize classifier
random_forest_clf = RandomForestClassifier()

# Train classifier
random_forest_clf.fit(X_train_transformed, y_train)


# ### F1 Score Random Forest Classifier without Filtering

# In[22]:


from sklearn.metrics import f1_score

y_pred = random_forest_clf.predict(X_test_transformed)

score = f1_score(y_test, y_pred, average='micro')
print(score)


# # End of Classifying Using PreProcessing only

# # Start of Classifying Using PreProcessing and Filtering tweets

# In[23]:


tweets = pd.read_csv('./Tweets.csv', encoding='utf-8')


# In[24]:


df_ = pd.DataFrame(tweets, columns=['text','airline_sentiment'])


# In[28]:


AirlineDataFrame = pd.DataFrame({}, columns=['text' , 'sentiment'])


# In[29]:


for index, row in df_.iterrows():
    text = row['text']
    if (not(re.search(r'\b'+'RT'+ r'\b',text))) and (detect(text) == 'en') and (len(text) > 20)  : 
        AirlineDataFrame = AirlineDataFrame.append(row)


# In[30]:


texts = AirlineDataFrame['text']
sentiment = AirlineDataFrame['airline_sentiment']


# In[31]:


# remove urls and strip
texts = [re.sub(r'https?:\/\/.*[\r\n]*', '', text).strip() for text in texts]
len(texts)


# In[32]:


# removing unordinal characters
texts = [''.join(character for character in text if ord(character) < 128) for text in texts]
len(texts)


# In[33]:


# remove punctuations
punctuation += "“’…”"
texts = [''.join(character for character in text.decode('utf-8').encode('ascii') if character not in punctuation) for text in texts]
len(texts)


# In[ ]:





# In[34]:


# tokenizing each text
tokenized = []
for text in texts:
    tokenized.append(word_tokenize(text))


# In[35]:


tokenized[0]


# In[36]:


# Stemming every token 
snowball_stemmer = nltk.stem.SnowballStemmer('english')
tokenized = [[snowball_stemmer.stem(word) for word in words] for words in tokenized]


# In[37]:


tokenized[0]


# In[38]:


# Removing every stopword
tokenized = [[word for word in words if word not in stopwords.words('english')] for words in tokenized]


# In[39]:


tokenized[0]


# In[40]:


# Appending each token forming a text
finalTexts = [' '.join(words) for words in tokenized]


# In[41]:


data_tuples = list(zip(finalTexts,sentiment))
df = pd.DataFrame(data_tuples , columns=['text', 'airline_sentiment'])
df.to_csv('FinalTweets.csv', sep='\t')


# In[ ]:





# In[42]:


X_train, X_test, y_train, y_test = train_test_split(finalTexts, sentiment, test_size=0.2, random_state=1)


# ## Applying TFIDF Vectorizer with Filtering

# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

# Calculating the tf-idf scores
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)


# In[44]:


print(X_train_transformed.shape)
print(X_test_transformed.shape)


# ## NaiveBayes Classifier with Filtering

# In[45]:


from sklearn.naive_bayes import MultinomialNB

# Initialize classifier
naive_bayes_clf = MultinomialNB()

# Train classifier
naive_bayes_clf.fit(X_train_transformed, y_train)


# In[46]:


print(naive_bayes_clf.score(X_test_transformed, y_test))


# ## F1 Score NaiveBayes with Filtering

# In[47]:


# F1 Score NaiveBayes
from sklearn.metrics import f1_score

y_pred = naive_bayes_clf.predict(X_test_transformed)

score = f1_score(y_test, y_pred, average='micro')
print(score)


# ## KNeighbors Classifier with Filtering

# In[48]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize classifier
k_nearest_clf = KNeighborsClassifier(n_neighbors=3)

# Train classifier
k_nearest_clf.fit(X_train_transformed, y_train)


# ## F1 Score KNeighbors with Filtering

# In[49]:


from sklearn.metrics import f1_score

y_pred = k_nearest_clf.predict(X_test_transformed)

score = f1_score(y_test, y_pred, average='micro')
print(score)


# ## Random Forest Classifier with Filtering

# In[50]:


from sklearn.ensemble import RandomForestClassifier

# Initialize classifier
random_forest_clf = RandomForestClassifier()

# Train classifier
random_forest_clf.fit(X_train_transformed, y_train)


# ## F1 Score Random Forest with Filtering

# In[51]:


from sklearn.metrics import f1_score

y_pred = random_forest_clf.predict(X_test_transformed)

score = f1_score(y_test, y_pred, average='micro')
print(score)


# # End of Classifying Using PreProcessing and Filtering tweets

# # Start of Classifying the SENTIMENT140 Dataset

# In[52]:


sentiment140 = pd.read_csv('./CompleteData.csv', engine='python', error_bad_lines=False)
sentiment140


# In[55]:


sentimentData = pd.DataFrame({}, columns=['text' , 'sentiment'])
sentimentData = sentiment140


# In[2]:


for index, row in sentimentDataFrame.iterrows():
    text = row['tweet_text']
    text = ''.join(character for character in text if ord(character) < 128)
    if (not(re.search(r'\b'+'RT'+ r'\b',text))) and (detect(text) == 'en') and (len(text) > 20)  : 
        sentimentData = sentimentData.append(row)


# In[56]:


len(sentimentData)


# ### Creating a new column representing the sentiment as a string
# ### 0 --> negative 2 --> neutral 4 --> positive

# In[57]:



sentimentData['sentiment_value'] = sentimentData['sentiment'].apply(lambda x: 'negative' 
if x==0 else ('neutral' if x==2 else 'positive'))
len(sentimentData)


# In[58]:


textsX = sentimentData['tweet_text']
sentimentY = sentimentData['sentiment_value']


# In[59]:


# remove urls and http
textsX = [re.sub(r'https?:\/\/.*[\r\n]*', '', text).strip() for text in textsX]


# In[60]:


# remove unordinal characters
textsX = [''.join(character for character in text if ord(character) < 128) for text in textsX]


# In[61]:


# remove punctuations
punctuation += "“’…”"
textsX = [''.join(character for character in text.decode('utf-8').encode('ascii') 
                  if character not in punctuation) for text in textsX]    
len(textsX)


# In[62]:


tokenizedSentiment140 = []
for text in textsX:
    tokenizedSentiment140.append(word_tokenize(text))
len(tokenizedSentiment140)


# In[65]:


snowball_stemmer = nltk.stem.SnowballStemmer('english')
tokenizedSentiment140 = [[snowball_stemmer.stem(word) for word in words] for words in tokenizedSentiment140]


# In[64]:


tokenizedSentiment140 = [[word for word in words if word not in stopwords.words('english')] 
                         for words in tokenizedSentiment140]


# In[66]:


finalTextsBonus = [' '.join(words) for words in tokenizedSentiment140]


# In[136]:


data_tuples = list(zip(finalTextsBonus,sentimentY))
df = pd.DataFrame(data_tuples , columns=['text', 'sentiment'])
df.to_csv('FinalTweetsTestBonus.csv', sep='\t')


# In[67]:


#df
print(len(finalTextsBonus))
print(len(sentimentY))


# In[ ]:





# In[68]:


Sentiment140X_train, Sentiment140X_test, Sentiment140y_train, Sentiment140y_test = train_test_split(finalTextsBonus, 
    sentimentY, test_size=0.2, random_state=1)


# ## Applying TFIDF Vectorizer Sentiment140 Dataset

# In[69]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

# Calculating the tf-idf scores
X_train_transformed = vectorizer.fit_transform(Sentiment140X_train)
X_test_transformed = vectorizer.transform(Sentiment140X_test)


# In[70]:


print(X_train_transformed.shape)
print(X_test_transformed.shape)


# ## NaiveBayes Classifier Sentiment140 Dataset

# In[71]:


from sklearn.naive_bayes import MultinomialNB

# Initialize classifier
naive_bayes_clf = MultinomialNB()

# Train classifier
naive_bayes_clf.fit(X_train_transformed, Sentiment140y_train)


# In[73]:


print(naive_bayes_clf.score(X_test_transformed, Sentiment140y_test))


# ## F1 Score NaiveBayes Sentiment140 Dataset

# In[74]:


# F1 Score NaiveBayes
from sklearn.metrics import f1_score

y_pred = naive_bayes_clf.predict(X_test_transformed)

score = f1_score(Sentiment140y_test, y_pred, average='micro')
print(score)


# ## KNeighbor Classifier Sentiment140 Dataset

# In[75]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize classifier
k_nearest_clf = KNeighborsClassifier(n_neighbors=3)

# Train classifier
k_nearest_clf.fit(X_train_transformed, Sentiment140y_train)


# ## F1 Score KNeighbor Sentiment140 Dataset

# In[ ]:


from sklearn.metrics import f1_score

y_pred = k_nearest_clf.predict(X_test_transformed)

score = f1_score(Sentiment140y_test, y_pred, average='micro')
print(score)


# ## Random Forest Classifier Sentiment140 Dataset

# In[148]:


from sklearn.ensemble import RandomForestClassifier

# Initialize classifier
random_forest_clf = RandomForestClassifier()

# Train classifier
random_forest_clf.fit(X_train_transformed, Sentiment140y_train)


# ## F1 Score Random Forest Sentiment140 Dataset

# In[149]:


from sklearn.metrics import f1_score

y_pred = random_forest_clf.predict(X_test_transformed)

score = f1_score(Sentiment140y_test, y_pred, average='micro')
print(score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




