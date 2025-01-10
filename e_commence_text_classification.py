# Import packages for assigning backend to use Tensorflow
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import packages
from sklearn.metrics import accuracy_score,f1_score
from sklearn import preprocessing,model_selection
from keras import layers

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import datetime
import pickle
import string
import json

# Import CSV file as Dataframe
e_com_df = pd.read_csv(os.path.join(os.getcwd(),r"datasets\ecommerceDataset.csv"),header=None,names=['category','text'])

# Drop duplicated found in the dataset
e_com_df = e_com_df.drop_duplicates()

# Drop row of data with null values
e_com_df = e_com_df.dropna()

# Remove punctuation string
e_com_df['text'] = e_com_df['text'].str.replace(f"[{string.punctuation}]", "", regex=True)

# Remove numeric string
e_com_df['text'] = e_com_df['text'].str.replace(r'\d+', '', regex=True)
# Convert all words into lowercase
e_com_df['text'] = e_com_df['text'].str.lower()

# Split the dataset into features and label
features = e_com_df['text'].values
label = e_com_df['category'].values

# Fit the label encoder with the label dataset
label_encoder = preprocessing.LabelEncoder()
label_encoded = label_encoder.fit_transform(label)

# Data splitting into train,test and validation dataset
SEED = 42
x_train,x_split,y_train,y_split = model_selection.train_test_split(features,label_encoded,train_size=0.7,random_state=SEED)
x_val,x_test,y_val,y_test = model_selection.train_test_split(x_split,y_split,train_size=0.5,random_state=SEED)

# Tokenization where the tokenizer are adapted from the train dataset
tokenizer = layers.TextVectorization(max_tokens=10000,output_sequence_length=200)
tokenizer.adapt(x_train)

# Embedding where the tokened words are converted into long vector 
embedding = layers.Embedding(10000,64)

# Model development where model layers are added depend on the function
model = keras.Sequential()
# NLP layers for tokenizaion and embedding
model.add(tokenizer)
model.add(embedding)
# RNN with Bidirectional LSRM are added with output layers match the number of category class
model.add(layers.Bidirectional(layers.LSTM(32,return_sequences=False)))
model.add(layers.Dense(len(e_com_df['category'].unique()),activation='softmax'))

# Compiling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Tensorboard logging callback function
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with callback from Tensorboard
model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=32,epochs=10, callbacks=[tensorboard_callback])

# Display the model architecture
model.summary()

# Get the prediction with the highest probabiltiy score
y_pred = np.argmax(model.predict(x_test),axis=1)

# Calculate the accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred,average='weighted')
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.2f}")

# Set the saved_models directory
os.makedirs("saved_models", exist_ok=True)

# Save the model as h5 file
model.save("saved_models/model.h5")

# Save the tokenizer as json file
vocab = tokenizer.get_vocabulary()
vocab_dict = {i: word for i, word in enumerate(vocab)}
with open("saved_models/tokenizer.json", "w") as f:
    json.dump(vocab_dict, f)

# Save the model as pkl file
with open("e_commence_encoder.pkl","wb") as f:
    pickle.dump(label_encoder,f)

# Instruct the user to run the command at terminal
print(f"Run `tensorboard --logdir={log_dir}` to view logs in TensorBoard.")