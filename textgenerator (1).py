#import dependancies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#data loading
from google.colab import files
uploaded=files.upload()
#load data
#loading data and opening our input data in the form of a txt file
file=open("chap2.txt").read()

#tokenisation 
#standardization
def tokenize_words(input):
  #lowercase everything to standardize it
  input=input.lower()
  #initiating the tokenizer
  tokenizer=RegexpTokenizer(r'\w+')
  #tokenize the txt into tokens
  tokens=tokenizer.tokenize(input)
  #filtering the stopwords using lambda
  filtered=filter(lambda token:token not in stopwords.words('english'),tokens)
  return "".join(filtered)
#preprocess the input data and make tokens
processed_inputs=tokenize_words(file)

#chars to numbers
#convert characters in our input to numbers
#we wil sort the list of the set of all characters that appear in our i/p txt and then use the enumerate fn to get numbers that represent characters
#we will then create a dictionary that stores the keys and values,or the characters and numbers that represent them
chars=sorted(list(set(processed_inputs)))
char_to_num=dict((c,i) for i,c in enumerate(chars))

#check if words to chars or chars to num(?!)has worked?

input_len=len(processed_inputs)
vocab_len=len(chars)
print("Total number of characters:",input_len)
print("Total vocab:",vocab_len)

#sequence length
seq_length=100
x_data=[]
y_data=[]

#loop through the sequence
for i in range(0,input_len - seq_length,1):
  in_seq=processed_inputs[i:i + seq_length]
  out_seq=processed_inputs[i+seq_length]
  x_data.append([char_to_num[char]for char in in_seq])
  y_data.append(char_to_num[out_seq])
n_patterns=len(x_data)
print("Total Patterns:",n_patterns)

#convert input sequence to np array and so on
X=numpy.reshape(x_data,(n_patterns,seq_length,1))
X=X/float(vocab_len)

#one-hot encoding
y=np_utils.to_categorical(y_data)

#creating the model
model=Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam')

#saving weights
filepath="model_weights_saved.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
desired_callbacks=[checkpoint]

#fit model and train
model.fit(X,y,epochs=4,batch_size=256,callbacks=desired_callbacks)

#recompile model with saved weights
filename="model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')

#output back into characters
num_to_char=dict((i,c) for i,c in enumerate(chars))

#random seed to help generate
start=numpy.random.randint(0,len(x_data)-1)
pattern=x_data[start]
print("Random Seed: ")
print("\"", ' '.join([num_to_char[value] for value in pattern]),"\"")