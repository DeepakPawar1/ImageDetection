# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:40:11 2018

@author: DELL
"""
import os
import os.path as path
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras import backend as K
MODEL_NAME = "heart_disease"

def load_data():
    data = pd.read_csv('C:\\Users\\DELL\\Desktop\\iris\\final_project\\Machine_Learning\\neew.csv')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,13].values
    from sklearn.model_selection import train_test_split 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    from sklearn.preprocessing import StandardScaler
    sc_X =StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train, y_train, X_test, y_test

def build_model():
    model = Sequential()
    model.add(Dense(output_dim = 7,init ='uniform',activation = 'relu',input_dim =13))
    model.add(Dense(output_dim = 7,init ='uniform',activation = 'relu'))
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
    
def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")
def main():
    if not path.exists('out'):
        os.mkdir('out')

    X_train, y_train, X_test, y_test = load_data()
    model = build_model()
    train(model, X_train, y_train, X_test, y_test)
    export_model(tf.train.Saver(), model, ["Age","Sex","ChestPain","BloodPressure","Cholestrol","Sugar","ExerciseSlope","Electrocardiographic","HeartRate","Exercise","Depression","MajorVessels","DefectType" ], "Heartdisease")
    
if __name__ == '__main__':
    main()