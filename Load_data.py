import os
from turtle import color
from cv2 import merge
import numpy as np #linear algebra
import csv   #data file 
import pandas as pd   #pandas for dataframe based data processingand cvs file i/o
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
import requests 
from bs4 import BeautifulSoup #for parsing and scraping html
import bs4
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *


# #making http requests 
# # url="https://merolagani.com/LatestMarket.aspx"
# url="https://nepsealpha.com/nepse-data"
# res=requests.get(url)
# # res.status_code
# res.json()
# content=res.json()
# content.keys()
# con=res.content
# print(type(con))
# # print(con)
# soup=BeautifulSoup(con,'html.parser') 
# # print(bcon.prettify)
# title=soup.title
# # print(type(title.string))
# # print(title.string)
# para=soup.find_all('p')


#reading the data
url="C:\\Users\\Hp\\Documents\\FYP\\Data\\nepse.csv"
df=pd.read_csv(url)
# print('Number of rows and columns:',df.shape)
# print(df.head())
# print(df.tail())
pd.merge()


#step:1(Create train data and test data)
training_data=df.iloc[1000:33438,:]
# print(training_data.head())
# print(training_data.shape)

testing_data=df.iloc[:1000,:]
# print(testing_data.tail())
# print(testing_data.shape)

#step(Drop the unncessary column)
training_data1=training_data.drop(['Date','Symbol','Percent Change'],axis=1)
# print(training_data.head())


#step:(Convert the data into standard format with a scaler,we will be using minmax scaler)
scaler=MinMaxScaler()
training_data1=scaler.fit_transform(training_data1)
# print(training_data.shape[0])

#step:(Converting our data into chunks i.e Taking previous few days of data as chunk and predicting the value of coming days)
X_train=[]
y_train=[]
for i in range(30,training_data1.shape[0]):
    X_train.append(training_data1[i-30:i])
    y_train.append(training_data1[i,3])

X_train,y_train=np.array(X_train),np.array(y_train)
# print(X_train.shape,y_train.shape)






# #step:(Build RNN Model/LSTM algorithm)
model=Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50,activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50,activation='relu', return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50,activation='relu', return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50,activation='relu'))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))
# print(X_train.shape[1],5)



# # Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])

# # Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 5, batch_size = 32)


data_30_days=training_data.tail(30)
d=data_30_days.append(testing_data,ignore_index=True)
d=d.drop(['Date','Symbol','Percent Change'],axis=1)
inputs=scaler.transform(d)

X_test=[]
y_test=[]
#step:(Create test data)
for i in range(30,inputs.shape[0]):
    X_test.append(training_data1[i-50:i])
    y_test.append(training_data1[i,3])


X_test,y_test=np.array(X_test),np.array(y_test)
print(X_test.shape)
# print(X_test)
# print(y_test)








#prediction
# y_pred=model.predict(X_test)
# # print(y_pred)
# y_pred= scaler.inverse_transform(np.asarray(y_pred))
y_test=scaler.inverse_transform(np.asarray(y_test))
print(y_test)

#step:(visualize the model)
# plt.figure(figsize=(20,6))
# plt.plot(y_test,color='blue')
# plt.plot(y_pred,color='green')
# plt.title('stock price prediction')
# plt.show()




