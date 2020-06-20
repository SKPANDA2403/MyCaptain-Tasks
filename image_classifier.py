#importing required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

#reading the required dataset and displating as as array
data=pd.read_csv('MNIST_data.csv')
data.head()
a=data.iloc[2,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)

#preparing the data
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

#creating test and train batches
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y, test_size=0.2,random_state=4)

#check data
x_train.head()

#calling the classifier
rf= RandomForestClassifier(n_estimators=100)

#fit the model
rf.fit(x_train,y_train)

#prediction
pred=rf.predict(x_test)

#printing predicted result and the type
print(pred)
print(type(pred))

#check prediction accuracy
s=y_test.values

#calculating correctly predicted values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count+=1
#printing out the required outputs
print("Total number of correct predictions made: "+count)
print("Total number of data present: "+len(pred))
accuracy=(count/len(pred))*100
print(f"Percentage Accuracy of the predictions: {accuracy}%")