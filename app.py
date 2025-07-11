import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 


# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('hamlet_nextword.keras')


# Function to predict next word
def predict_next_word(text, model, tokenizer, max_sequence_length):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequence_length:
    token_list = token_list[-(max_sequence_length-1):]
  token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
  predicted = model.predict(token_list, verbose=0)
  predicted_word_index = np.argmax(predicted,axis=1)
  for word ,index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None


# Streamlit app
st.title("Next Word Prediction Using LSTM")
input_text = st.text_input("Enter a sentence to predict the next word:","To be or not to be")
if st.button("Predict Next Word"):
    max_sequence_lenght = model.input_shape[1] + 1
    next_word = predict_next_word(input_text, model, tokenizer, max_sequence_lenght)
    st.write(f"The Next word is : {next_word}")
