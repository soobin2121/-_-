#!/usr/bin/env python
# coding: utf-8

# # TDM 직접구현

# ### 1. 공백으로 토큰화

# In[1]:


docs = ['동물원 코끼리','동물원 원숭이 바나나','엄마 코끼리 아기 코끼리','원숭이 바나나 코끼리 바나나']


# In[2]:


doc_1s=[]
for doc in docs:
    doc_1s.append(doc.split(' '))
    
doc_1s    


# ### 2. 토큰 배열화

# In[16]:


from collections import defaultdict
word2id = defaultdict(lambda : len(word2id))

for doc in doc_1s:
    for token in doc:
        word2id[token]
        
word2id        


# In[17]:


doc_1s


# In[18]:


doc


# In[5]:


import numpy as np

TDM = np.zeros((len(word2id), len(doc_1s)), dtype=int)      #6 by 4
print(TDM)

for i, doc in enumerate(doc_1s):
    for token in doc:
        TDM[word2id[token], i] +=1
        
TDM        


# In[8]:


import pandas as pd

doc_names = ['문서'+ str(i) for i in range(len(doc_1s))]
print('doc_names', doc_names)


sorted_vocab = sorted((value, key) for key, value in word2id.items())
vocab = [v[1] for v in sorted_vocab]

df_TDM = pd. DataFrame(TDM, columns = doc_names)
df_TDM['단어'] = vocab
df_TDM.set_index('단어')


# In[ ]:


## sklearn


# In[9]:


docs = ['동물원 코끼리','동물원 원숭이 바나나','엄마 코끼리 아기 코끼리','원숭이 바나나 코끼리 바나나']ㅜ


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
DTM = count_vect.fit_transform(docs)
DTM.toarray()


# In[15]:


import pandas as pd

doc_names = ['문서'+ str(i) for i in range(len(doc_1s))]
vocab = count_vect.get_feature_names()
print(vocab)



df_TDM = pd. DataFrame(DTM.toarray().T ,columns = doc_names)
df_TDM['단어'] = vocab
df_TDM.set_index('단어')

