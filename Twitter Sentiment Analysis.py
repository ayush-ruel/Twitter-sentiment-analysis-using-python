#!/usr/bin/env python
# coding: utf-8

# In[19]:


import re


# In[5]:


import nltk  # for text manipulation 
import string 
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  


# In[6]:


pd.set_option("display.max_colwidth", 200) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


train  = pd.read_csv('Downloads/train.csv') 
test = pd.read_csv('Downloads/test_tweets.csv')


# In[9]:


train[train['label'] == 0].head(10)


# In[10]:


train[train['label'] == 1].head(10)


# In[11]:


train.shape, test.shape


# In[12]:


train["label"].value_counts()


# In[13]:


length_train = train['tweet'].str.len() 
length_test = test['tweet'].str.len() 
plt.hist(length_train, bins=20, label="train_tweets") 
plt.hist(length_test, bins=20, label="test_tweets") 
plt.legend() 
plt.show()


# In[14]:


#Combining train and test datasets for easy operations. Will split it back afterwards. 
combi = train.append(test, ignore_index=True) 
combi.shape


# In[16]:


#Removing unwanted text patterns
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


# In[20]:


#Removing any word starting with '@'
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 
combi.head()


# In[21]:


#Replacing everything accept characters and '#' with spaces.
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") 
combi.head(10)


# In[22]:


#Removing all words with length <=3
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combi.head()


# In[23]:


#Tokenizing
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) 
tokenized_tweet.head()


# In[24]:


#Normalizing the tweets
from nltk.stem.porter import * 
stemmer = PorterStemmer() 
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming


# In[25]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    
combi['tidy_tweet'] = tokenized_tweet


# In[23]:


get_ipython().system('pip install wordcloud ')


# In[26]:


from wordcloud import WordCloud


# In[27]:


#Wordcloud for all the tweets
all_words = ' '.join([text for text in combi['tidy_tweet']])  
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()


# In[28]:


#Words in non racist/sexist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()


# In[29]:


#Racist/sexist tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()


# In[30]:


# function to collect hashtags
def hashtag_extract(x):    
    hashtags = []    # Loop over the words in the tweet    
    for i in x:        
        ht = re.findall(r"#(\w+)", i)        
        hashtags.append(ht)     
    return hashtags


# In[31]:


# extracting hashtags from non racist/sexist tweets 
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0]) 

# extracting hashtags from racist/sexist tweets 
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1]) 

# unnesting list

HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

# print the hashtags
HT_regular, HT_negative


# In[32]:


#plotting a bar graph for top 10 hashtags in non racist tweets
a = nltk.FreqDist(HT_regular) 
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())}) 

# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5)) 
ax = sns.barplot(data=d, x= "Hashtag", y = "Count") 
ax.set(ylabel = 'Count') 
plt.show()


# In[33]:


#plotting a bar graph for top 10 hashtags in racist tweets
b = nltk.FreqDist(HT_negative) 
e = pd.DataFrame({'Hashtag': list(b.keys()),'Count': list(b.values())})

# selecting top 20 most frequent hashtags 
e = e.nlargest(columns="Count", n = 20)   
plt.figure(figsize=(16,5)) 
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")


# In[ ]:




