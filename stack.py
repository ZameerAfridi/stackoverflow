#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


# In[2]:


train_data= pd.read_csv("D:/arg_assignments/Project_ml/Train.csv", nrows=1000000)


# In[3]:


train_data


# In[4]:


train_data.drop(columns=["Id"],axis=1,inplace=True)


# In[5]:


train_data


# In[6]:


print(train_data.isnull().sum())


# In[7]:


train_data.dropna(inplace=True)


# In[8]:


print(train_data.isnull().sum())


# In[9]:


train_data["Count_tags"]=train_data.Tags.apply(lambda x:len(str(x).split()))


# In[10]:


train_data


# In[10]:


sns.countplot(train_data.Count_tags)
plt.title("Number of Tags associated with the quarie")
plt.xlabel("Number of Tags")
plt.ylabel("Number of Quries")


# In[11]:


tags_vector=CountVectorizer(tokenizer = lambda x: x.split(),binary=True)
x_tags=tags_vector.fit_transform(train_data.Tags)
count=x_tags.sum(axis=0)
freq_tags=pd.DataFrame()
freq_tags["tags"]=tags_vector.get_feature_names()
freq_tags["count"]=count.tolist()[0]
freq_tags.sort_values(by='count',ascending=False,inplace=True)
freq_tags.reset_index(inplace=True)
freq_tags.drop(columns="index",inplace=True)
print(freq_tags)


# In[12]:


print("----------------------The Top 10 Tags in the Corpus ---------------")
print(freq_tags["tags"][:10].values)
print("--------------------Top 10 Least Frequent tags in the Corpus--------------")
print(freq_tags.tail(10)["tags"].values)


# In[13]:


sns.set_style("whitegrid")
sns.lineplot(data=freq_tags,x=freq_tags.index,y="count")
plt.title("Occurence of Tags in corpus",color="blue")


# In[14]:


sns.set_style("whitegrid")
sns.lineplot(data=freq_tags.iloc[:1000],x=freq_tags.index[:1000],y="count")
plt.title("Occurence of Tags in corpus",color="blue")


# In[15]:


plt.figure(figsize=(10,7))
sns.set_style("whitegrid")
sns.lineplot(data=freq_tags.iloc[:100],x=freq_tags.index[:100],y="count")
a=[0,20,40,60,80,100]
sns.scatterplot(a,freq_tags.iloc[a,1],hue=freq_tags.iloc[a,0])
plt.title("Occurence of Tags in corpus",color="blue")


# In[16]:


t=freq_tags.tags.to_list()
c=freq_tags["count"].to_list()
tags_dic={}
for i,j in zip(t,c):
    tags_dic[i]=j
#print(tags_dic)
wordcloud = WordCloud().generate_from_frequencies(tags_dic)
plt.figure(figsize=(8,8))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[17]:


## Data preprocessing
#part-1 Defining Meta Feature before Preprocessing(Experimental)
#Number of Characters in Title
#Number of Character in Body column
#Number of Code Segemts in each row
#Number of Words in Title
char_title_pre=train_data.Title.apply(lambda x:len(x))
char_body_pre=train_data.Body.apply(lambda x:len(x))
code_body_pre=train_data.Body.apply(lambda x:len(re.findall(r"<code>",x)))
words_title_pre=train_data.Title.apply(lambda x:len(str(x).split()))


# In[18]:


# Defining Some Extra StopWords seems to be Not useful
extra=['could',"would","iis","sometimes","sometime","puts","put","get","gets","help","please","need",       "like","know","thank","thanks","madam","sir","hii","doubt","doubts","www","com"]
li=stopwords.words("english")
li=li+extra


# In[19]:


## Preprocessing
##  Removing html tags,urls
##  Removing the code segemts (assumes codes are varies much)
#   Removing the Punctuations and Numbers and Keeping only text

def Preprocesser(doc):
    body=[]
    for text in doc:
        text=re.sub(r"href.*","",text)          #removing the href ie. removing the hyper links
        text=re.sub('<code>(.*?)</code>', '', text, flags=re.MULTILINE|re.DOTALL) # removing the code segments
        text=re.sub('<.*?>', ' ', str(text.encode('utf-8')))           #removing the Html Tags in the Text
        text=re.sub(r"[^a-zA-Z]+"," ",text)  ## removing numbers and most of Puncutuations in the Text.
        text=text.lower()                    ## converting from upper case to lower case
        body.append(" ".join([k for k in text.split() if((len(k)>2 or k=="c") and k not in li )]))
        
    return body


# In[20]:


pre_body=Preprocesser(train_data.Body)      ## Preprocessing the Body Columns


# In[21]:


pre_text=Preprocesser(train_data.Title)     ## Preprocessing the Title Columns


# In[22]:


## Replacing the a Title and Body with the Preprocessed Title and Body respectively.
train_data["Title"]=pre_text
train_data["Body"]=pre_body
pre_data=train_data    # Creating a Reference for the train_data_100k 


# In[23]:


pre_data


# In[25]:


## part 2: Defining Meta Features After Preprocessing the Dataset
##    Number of words in the Title
##    Number of Words in the Body
words_title_post=train_data.Title.apply(lambda x:len(str(x).split()))
words_body_post=train_data.Body.apply(lambda x:len(str(x).split()))


# In[26]:


##  Adding to the Dataset
##   Adding the meta Feautures to the train dataset and normalize the Features.


# In[27]:


train_data["char_title_pre"]=(char_title_pre-min(char_title_pre))/(max(char_title_pre)-min(char_title_pre))
train_data["char_body_pre"]=(char_body_pre-min(char_body_pre))/(max(char_body_pre)-min(char_body_pre))
train_data["code_body_pre"]=(code_body_pre-min(code_body_pre))/(max(code_body_pre)-min(code_body_pre))
train_data["words_title_pre"]=(words_title_pre-min(words_title_pre))/(max(words_title_pre)-min(words_title_pre))
train_data["words_title_post"]=(words_title_post-min(words_title_post))/(max(words_title_post)-min(words_title_post))
train_data["words_body_post"]=(words_body_post-min(words_body_post))/(max(words_body_post)-min(words_body_post))


# In[28]:


train_data.head()


# In[29]:


## seperate the tag columns and the droping the Tags column from the dataset
y_tagss=train_data.Tags
train_data.drop(columns="Tags",axis=1,inplace=True)


# In[30]:


pre_data=train_data
pre_data


# In[ ]:


# Converting the Tags columns in to the Mulit label Classification
# initializing the Count vectorizer
tag_vect=CountVectorizer(binary=True,tokenizer=lambda x:str(x).split(),max_features=500)
vec_tag=tag_vect.fit_transform(y_tagss)


# In[38]:


## split the Training dataset and validataion dataset
x_train,x_val,y_train,y_val=train_test_split(pre_data.iloc[:500000,],vec_tag[:500000],test_size=0.2)
print("The shape of the Training Dataset :",x_train.shape,y_train.shape)
print("The shape of the validation Dataset :",x_val.shape,y_val.shape)


# In[39]:


###############################################################################################
###                     Model Building       ###


# In[40]:


## Concatenating Title And Body to a Single Feature
tit_bod_train=[i+" "+j for i,j in zip(x_train.Title,x_train.Body)] ## combining the both the title and Body
tit_bod_val=[i+" "+j for i,j in zip(x_val.Title,x_val.Body)]  # combining the title and Body for the Validation


# In[44]:


feat_vec=TfidfVectorizer(tokenizer=lambda x:x.split(),max_features=100000,ngram_range=(1,1))
feat_vec.fit(tit_bod_train)
train_feat=feat_vec.transform(tit_bod_train)
val_feat=feat_vec.transform(tit_bod_val)


# In[48]:


# Concatenate the Title_Body and Derived Features 
train_feat=hstack((train_feat,x_train.iloc[:,2:].values))
val_feat=hstack((val_feat,x_val.iloc[:,2:].values))


# In[50]:


print("The Shape of the Training Dataset :",train_feat.shape)
print("The Shape of the Validation Dataset :",val_feat.shape)


# In[57]:


### Using the Log Loss (Linear MOdels --> Logistic regression)

classifier = OneVsRestClassifier(SGDClassifier(penalty='l2',loss="log",alpha=0.000001), n_jobs=-1)
classifier.fit(train_feat, y_train)
val_pre = classifier.predict(val_feat)

print("accuracy :",metrics.accuracy_score(y_val,val_pre))
print("macro f1 score :",metrics.f1_score(y_val, val_pre, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_val, val_pre, average = 'micro'))


# In[58]:


## Hinge Loss ---> Linear SVM Classifier 
classifier = OneVsRestClassifier(SGDClassifier(penalty='l2',loss="hinge",alpha=0.000001), n_jobs=-1)
classifier.fit(train_feat, y_train)
val_pre = classifier.predict(val_feat)

##VALIDATION ACCURACY
print("accuracy :",metrics.accuracy_score(y_val,val_pre))
print("macro f1 score :",metrics.f1_score(y_val, val_pre, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_val, val_pre, average = 'micro'))


# In[59]:


## TRAINING ACCURACY
val_pre = classifier.predict(train_feat)

print("accuracy :",metrics.accuracy_score(y_train,val_pre))
print("macro f1 score :",metrics.f1_score(y_train, val_pre, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_train, val_pre, average = 'micro'))


# In[ ]:


## ‘modified_huber Loss’---> Problastic Models 
classifier = OneVsRestClassifier(SGDClassifier(penalty='l2',loss='modified_huber',alpha=0.0000001), n_jobs=-1)
classifier.fit(train_feat, y_train)
val_pre = classifier.predict(val_feat)

print("accuracy :",metrics.accuracy_score(y_val,val_pre))
print("macro f1 score :",metrics.f1_score(y_val, val_pre, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_val, val_pre, average = 'micro'))


# In[ ]:




