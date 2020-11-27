#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install emoji')


# ### Working with emoji package



import emoji as emoji




# emoji.EMOJI_UNICODE --> to see all thr emojis




emoji_dictionary = {"0": "\u2764\uFE0F",    
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                    "5": ":hundred_points:",
                    "6": ":fire:",
                    "7": ":chestnut:"
                   }




emoji.emojize(":chestnut:")




for e in emoji_dictionary.values():           # print all emoji in the list
   # print(emoji.emojize(e))


# ### Step2: Processing custom emoji dataset



import numpy as np
import pandas as pd




train = pd.read_csv('train_emoji.csv',header=None)
test = pd.read_csv('test_emoji.csv',header=None)




train.head()




# Let us print the sentences with emojis 
data = train.values
# print(data.shape)



XT = train[0]
Xt = test[0]

YT = train[1]
Yt = test[1]




for i in range(5):
    print(XT[i],emoji.emojize(emoji_dictionary[str(YT[i])]))


# ### Step 3: Converting sentences into Embeddings



embeddings = {}
with open('glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        
        embeddings[word] = coeffs




embeddings["eat"]




emb_dim = embeddings["eat"].shape[0]
# print(emb_dim)


# ### Step  4: Converting Sentences into Vector (Embedding layer output)



def getOutputEmbeddings(X):
    
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]
            
    return embedding_matrix_output



emb_XT = getOutputEmbeddings(XT)
emb_Xt = getOutputEmbeddings(Xt)




# print(emb_XT.shape)
# print(emb_Xt.shape)




from keras.utils import to_categorical




YT=to_categorical(YT,num_classes=5)
Yt=to_categorical(Yt,num_classes=5)


# ### Step 5: Define the RNN/LSTM Model



from keras.layers import *
from keras.models import Sequential


# creating a stacked LSTM Model

model = Sequential()
model.add(LSTM(64,input_shape=(10,50),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()




from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

checkpoint= ModelCheckpoint("best_model.h5",monitor='val_loss',verbose=True,save_best_only=True)
earlystop= EarlyStopping(monitor='val_acc',patience=10)

hist = model.fit(emb_XT,YT,batch_size=32,epochs=100 ,shuffle=True,validation_split=0.1)




pred = model.predict_classes(emb_Xt)




# print(pred)



model.evaluate(emb_Xt,Yt)




pred = model.predict_classes(emb_Xt)



# visulaizing the emoji pridictions 
for i in range(30):
    print(' '.join(Xt[i]))
    print(emoji.emojize(emoji_dictionary[str(np.argmax(Yt[i]))]))
    print(emoji.emojize(emoji_dictionary[str(pred[i])]))
