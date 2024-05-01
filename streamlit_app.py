import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

review_len = 250
(train_reviews, train_labels), (test_reviews, test_labels) = imdb.load_data(maxlen = review_len)
(train_reviews_bak, train_labels_bak), (test_reviews_bak, test_labels_bak) = imdb.load_data(maxlen = review_len)

train_reviews = pad_sequences(train_reviews, padding='post')
test_reviews = pad_sequences(test_reviews, padding='post')
train_labels = keras.utils.to_categorical(train_labels, 2)
test_labels = keras.utils.to_categorical(test_labels, 2)

index = 0
print(train_reviews[index])
print(train_labels[index])

word_index = imdb.get_word_index()
#inverted_word_index = dict((i, word) for (word, i) in word_index)
word_index['great']

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
model = Sequential()
vocab_size = len(word_index)
embedding_dim = 16
max_size = review_len -1
model.add(Embedding(vocab_size, embedding_dim, input_length = max_size))
model.add(GlobalAveragePooling1D())

model.add(Dense(14, activation = 'sigmoid'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

model.fit(train_reviews, train_labels, epochs=10, shuffle = True)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token='<OOV>')
tokenizer.word_index = word_index

sentences = [
     st.text_input('Enter some text')
    
 ]
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post', maxlen = review_len -1)

index = 0
sample = padded[index].reshape(1, len(padded[index]))
prediction = model.predict(sample)

st.write(prediction)
