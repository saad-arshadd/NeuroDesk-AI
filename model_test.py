
import json
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Load trained model
model = keras.models.load_model("Chat_model.h5")

# Load tokenizer and label encoder
with open("tokenizer.pkl", "rb") as f:#Savng the tokenixer so  we can preprocess future inputs the exact same way as during training.
    tokenizer = pickle.load(f)# to load the trained tokenizer file

with open("label_encoder.pkl", "rb") as encoder_file:#Serializes and saves the label encoder, so you can convert predictions back to readable text labels later.
    label_encoder = pickle.load(encoder_file)

# Chat 

while True:
    input_text = input("Enter your command -> ")
    padded_sequences = pad_sequences(tokenizer.texts_to_sequences([input_text]), maxlen=50, truncating='post')
    result = model.predict(padded_sequences)
    tag = label_encoder.inverse_transform([np.argmax(result)])
##and now based on this tag we  have to take out our response
    for i in data['intents']:
        if i['tag'] == tag[0]:  # Fix here #means coming from same greeting from above line 36
            print(np.random.choice(i['responses'])) #then if from above tag then choose any response from the data










