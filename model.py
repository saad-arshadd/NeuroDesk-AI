 #.\tf_env\Scripts\activate  
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

with open("intents.json") as file:
    data=json.load(file)

training_sentence=[]
training_labels=[]
labels=[]
responses=[]

#x are the seentcnes x and y(tags)are their labels , jinper ham train karenge 
#for example x=how are you and its y=greeting etc

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentence.append(pattern) #x feature for ttraining
        training_labels.append(intent['tag']) # y for the trainingg
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
number_of_classes=len(labels)

print(number_of_classes)

label_encoder = LabelEncoder()
label_encoder.fit(training_labels) # to convert it into number , tags like 1 ,greeting -1  goofbye 2 ..etc
training_labels= label_encoder.transform(training_labels)

vocab_size=1000
ovv_token = "<OOV>" #Stands for "Out Of Vocabulary" token. Any word not seen during training will be replaced with this token (e.g., <OOV>).
max_len=50
embedding_dim=16



tokenizer=Tokenizer(num_words  =vocab_size, oov_token=ovv_token)
tokenizer.fit_on_texts(training_sentence) #Builds the vocabulary from the training_sentence list. Assigns each word a unique index.
word_index=tokenizer.word_index #Stores a dictionary mapping each word → its unique index, based on frequency (most common word gets lowest index).
sequences = tokenizer.texts_to_sequences(training_sentence) # Converts each sentence into a list of integers using word_index.
padded_sequences = pad_sequences(sequences,truncating='post',maxlen=max_len) #Ensures all sequences are exactly 50 tokens long and If a sentence is shorter than 50, zeros are added at the beginning (can be changed with padding='post')

model=Sequential()
model.add(Embedding(vocab_size ,embedding_dim,input_length=max_len )) #Converts each word index into a dense vector of fixed size (learned during training)


# 8th july 2024

model.add(GlobalAveragePooling1D())
model.add(Dense(16,activation="relu")) # relu means non linear
model.add(Dense(16,activation="relu"))
model.add(Dense(number_of_classes,activation="softmax")) #softmax gives a probablity distribuation accross al the classess

model.compile(loss='sparse_categorical_crossentropy',optimizer="adam",metrics=["accuracy"])

model.summary()


history =model.fit(padded_sequences , np.array(training_labels),epochs=1000) # start trainng model , more epochs mean greater accuracy but can also lead to over fitting

model.save("Chat_model.h5")
with open ("tokenizer.pkl","wb") as f: #Savng the tokenixer so  we can preprocess future inputs the exact same way as during training.
    pickle.dump(tokenizer,f,protocol=pickle.HIGHEST_PROTOCOL)



with open ("label_encoder.pkl","wb") as enocder_file: #Serializes and saves the label encoder, so you can convert predictions back to readable text labels later.
    pickle.dump(label_encoder,enocder_file,protocol=pickle.HIGHEST_PROTOCOL)


        
        
    