# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:40:11 2018

@author: DELL
"""
import os
import os.path as path
import pandas as pd
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K
MODEL_NAME = "heart_disease"
datae = pd.read_csv('C:\\Users\\DELL\\Desktop\\iris\\final_project\\Machine_Learning\\neew.csv')

def load_data():
    data = pd.read_csv('C:\\Users\\DELL\\Desktop\\iris\\final_project\\Machine_Learning\\neew.csv')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,13].values
   
    from sklearn.model_selection import train_test_split 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    
    return X_train, y_train, X_test, y_test

def build_model():
    model = Sequential()
    model.add(Dense(output_dim = 11,init ='uniform',activation = 'relu',input_dim =13))
    model.add(Dense(output_dim = 11,init ='uniform',activation = 'relu'))
    model.add(Dense(output_dim = 11,init ='uniform',activation = 'relu'))
    model.add(Dense(output_dim = 1,init ='uniform',activation = 'sigmoid'))
    return model

def train(model, X_train, y_train, X_test, y_test):
    model.compile(loss='binary_crossentropy', \
                  optimizer='adam', \
                  metrics=['accuracy'])

    model.fit(X_train, y_train, \
              batch_size=10, \
              epochs=100, \
              verbose=1, \
              validation_data=(X_test, y_test))
def bloodpressure(m):
    
    a = set()
    b =[]
    c=[]
    d=[]
    e=[]
    for i in range(1,217):
        
        if(datae.Age[i] >= 28 & datae.Age[i] <=40):
            b.append(datae.BloodPressure[i])
        if(datae.Age[i]>40 & datae.Age[i] <=55):
            c.append(datae.BloodPressure[i])
        if(datae.Age[i]>55 & datae.Age[i] <=65):
            d.append(datae.BloodPressure[i])
        if(datae.Age[i]>65 & datae.Age[i] <=90):
            e.append(datae.BloodPressure[i])
        else:
            break
    for a in range(1):
        if(m>28 & m <40):
            return sum(b)/(len(b))
        elif(m>40 & m<55):
            return sum(c)/(len(c))
        elif(m>55 & m <65):
            return sum(d)/(len(d))
        elif(m>65 & m <90):
            return sum(e)/(len(e))
        else:
            break              
def cholestrol(m):
    
    a = set()
    b =[]
    c=[]
    d=[]
    e=[]
    for i in range(1,217):
        
        if(datae.Age[i] >= 28 & datae.Age[i] <=40):
            b.append(datae.Cholestrol[i])
        if(datae.Age[i]>40 & datae.Age[i] <=55):
            c.append(datae.Cholestrol[i])
        if(datae.Age[i]>55 & datae.Age[i] <=65):
            d.append(datae.Cholestrol[i])
        if(datae.Age[i]>65 & datae.Age[i] <=90):
            e.append(datae.Cholestrol[i])
        else:
            break
    for a in range(1):
        if(m>28 & m <40):
            return sum(b)/(len(b))
        elif(m>40 & m<55):
            return sum(c)/(len(c))
        elif(m>55 & m <65):
            return sum(d)/(len(d))
        elif(m>65 & m <90):
            return sum(e)/(len(e))
        else:
            break
def HeartRate(m):
    
    a = set()
    b =[]
    c=[]
    d=[]
    e=[]
    for i in range(1,217):
        
        if(datae.Age[i] >= 28 & datae.Age[i] <=40):
            b.append(datae.HeartRate[i])
        elif(datae.Age[i]>40 & datae.Age[i] <=55):
            c.append(datae.HeartRate[i])
        elif(datae.Age[i]>55 & datae.Age[i] <=65):
            d.append(datae.HeartRate[i])
        elif(datae.Age[i]>65 & datae.Age[i] <=90):
            e.append(datae.HeartRate[i])
        else:
            break
    for a in range(1):
        if(m>28 & m <40):
            return sum(b)/(len(b))
        elif(m>40 & m<55):
            return sum(c)/(len(c))
        elif(m>55 & m <65):
            return sum(d)/(len(d))
        elif(m>65 & m <90):
            return sum(e)/(len(e))
        else:
            break
def Depression(m):
    
    a = set()
    b =[]
    c=[]
    d=[]
    e=[]
    for i in range(1,217):
        
        if(datae.Age[i] >= 28 & datae.Age[i] <=40):
            b.append(datae.Depression[i])
        elif(datae.Age[i]>40 & datae.Age[i] <=55):
            c.append(datae.Depression[i])
        elif(datae.Age[i]>55 & datae.Age[i] <=65):
            d.append(datae.Depression[i])
        elif(datae.Age[i]>65 & datae.Age[i] <=90):
            e.append(datae.Depression[i])
        else:
            break
    for a in range(1):
        if(m>28 & m <40):
            return sum(b)/(len(b))
        elif(m>40 & m<55):
            return sum(c)/(len(c))
        elif(m>55 & m <65):
            return sum(d)/(len(d))
        elif(m>65 & m <90):
            return sum(e)/(len(e))
        else:
            break
def MajorVessels(m):
    
    a = set()
    b =[]
    c=[]
    d=[]
    e=[]
    for i in range(1,217):
        
        if(datae.Age[i] >= 28 & datae.Age[i] <=40):
            b.append(datae.MajorVessels[i])
        elif(datae.Age[i]>40 & datae.Age[i] <=55):
            c.append(datae.MajorVessels[i])
        elif(datae.Age[i]>55 & datae.Age[i] <=65):
            d.append(datae.MajorVessels[i])
        elif(datae.Age[i]>65 & datae.Age[i] <=90):
            e.append(datae.MajorVessels[i])
        else:
            break
    for a in range(1):
        if(m>28 & m <40):
            return sum(b)/(len(b))
        elif(m>40 & m<55):
            return sum(c)/(len(c))
        elif(m>55 & m <65):
            return sum(d)/(len(d))
        elif(m>65 & m <90):
            return sum(e)/(len(e))
        else:
            break
def DefectType(m):
    
    a = set()
    b =[]
    c=[]
    d=[]
    e=[]
    for i in range(1,217):
        
        if(datae.Age[i] >= 28 & datae.Age[i] <=40):
            b.append(datae.DefectType[i])
        elif(datae.Age[i]>40 & datae.Age[i] <=55):
            c.append(datae.DefectType[i])
        elif(datae.Age[i]>55 & datae.Age[i] <=65):
            d.append(datae.DefectType[i])
        elif(datae.Age[i]>65 & datae.Age[i] <=90):
            e.append(datae.DefectType[i])
        else:
            break
    for a in range(1):
        if(m>28 & m <40):
            return sum(b)/(len(b))
        elif(m>40 & m<55):
            return sum(c)/(len(c))
        elif(m>55 & m <65):
            return sum(d)/(len(d))
        elif(m>65 & m <90):
            return sum(e)/(len(e))
        else:
            break

def user():
    a = []
    a.append(int(input("What is your age")) )  
    a.append(int(input("gender 1 for male 0 for female")))
    a.append(int(input("did you have a chestpain if yes please give the number of times you had it")))
    nj= int(input("Do you know the blood pressure 1 for yes 0 for no"))
    if nj == 0:
        k=bloodpressure(a[0])
        a.append(k)
    else:
        a.append(nj)
    lk= int(input("Do you know the colesttrol typeit if you know or enter  0 "))
    if lk == 0:
       a.append(cholestrol(a[0]))
    else:
        a.append(lk)
    a.append(int(input("enter 1 if you have sugar else enter 0")))
    a.append(2)
    kj= int(input("Do you know the heart rate type it if you know or enter  0"))
    if kj == 0:
        a.append(HeartRate(a[0]))
    else:
        a.append(kj)
    a.append(int(input("Do yo exercise 1 for yes and 0 for no")))
    oj= int(input("Do you know the depression level type it if you know or enter  0"))
    if oj == 0:
        a.append(Depression(a[0]))
    else:
        a.append(oj)
    a.append(np.random.randint(1,3))
    a.append(MajorVessels(a[0])) 
    a.append(DefectType(a[0])) 
    print("the a list is", a) 
    a = np.array(a).reshape(1,13)
    return a
        

def predict(model,a):
    ypred = model.predict(a)
    print(ypred)
    

def main():
    if not path.exists('out'):
        os.mkdir('out')

    X_train, y_train, X_test, y_test = load_data()
    model = build_model()
    train(model, X_train, y_train, X_test, y_test)
    a = user()
    predict(model,a)
    print(np.shape(X_test))
if __name__ == '__main__':
    main()