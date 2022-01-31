# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:59:25 2021

@author: Luciano
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime 
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

class Data(object):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        

def runRegression(dataSetByInvoice, customerId):
    """
    Dado un determinado usuario, se procedera a dividir sus pedidos
    2/3 sera utilizado para entrenar el modelo (X_train, y_train)
    1/3 sera utilizado para predecir.(X_test, Y_test)
    retornando estos valores mas la predicción de 
    tanto del set de entrenamiento como el set de testeo.
    """
      
    selectClient = dataSetByInvoice.customerId == int(customerId)
    ordersByClient = dataSetByInvoice[selectClient]
    print(ordersByClient)
    X_train, X_test, Y_train, Y_test = train_test_split(ordersByClient['products'].values, ordersByClient['totalValue'].values, test_size=1/3, random_state=0)

    X_train = pd.to_numeric(X_train).reshape(-1, 1)
    X_test = pd.to_numeric(X_test).reshape(-1, 1)
    Y_train = pd.to_numeric(Y_train).reshape(-1, 1)
    Y_test = pd.to_numeric(Y_test).reshape(-1, 1)

    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    
    # regresión lineal 
    # del set de entrenamiento y predicion de la misma
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    return  X_train, X_test, Y_train, Y_test, y_train_pred , y_test_pred

def runRegressionByMonth(dataSetByInvoice, client):
    """
    Dado un determinado usuario, se procedera a dividir sus pedidos por mes calendario (formato gregoriano)
    2/3 sera utilizado para entrenar el modelo (X_train, y_train)
    1/3 sera utilizado para predecir.(X_test, Y_test)
    retornando estos valores mas la predicción de 
    tanto del set de entrenamiento como el set de testeo.
    """
    
    orderByClient = dataSetByInvoice.customerId == int(client)
    
    ordersBymonth=dataSetByInvoice[orderByClient].groupby(dataSetByInvoice.invoiceDate.dt.strftime('%Y-%m'), as_index = True)['totalValue'].sum().sort_index(ascending = True)
    
    dateGregorian = ordersBymonth.index.map(lambda t: pd.to_datetime(t).toordinal())
    X_train, X_test, Y_train, Y_test = train_test_split(dateGregorian, ordersBymonth.values, test_size=1/3, random_state=0)
  
    X_train=  pd.to_numeric(X_train.values).reshape(-1, 1)
    X_test =  pd.to_numeric(X_test.values).reshape(-1, 1)
    Y_train = pd.to_numeric(Y_train).reshape(-1, 1)
    Y_test = pd.to_numeric(Y_test).reshape(-1, 1)
    
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    
    # regresión lineal 
    # del set de entrenamiento y predicion de la misma
    y_train_pred= regressor.predict(X_train)
    y_test_pred= regressor.predict(X_test)
    
    return X_train, X_test, Y_train, Y_test, y_train_pred , y_test_pred 

def runRegressionByOrder(ordersByClient, column, row):
    """
    Dado un determinado usuario, se procedera una regresión sobre cada pedido 
    2/3 sera utilizado para entrenar el modelo (X_train, y_train)
    1/3 sera utilizado para predecir.(X_test, Y_test)
    retornando estos valores mas la predicción de 
    tanto del set de entrenamiento como el set de testeo.
    """
        
     
    X_train, X_test, Y_train, Y_test = train_test_split(ordersByClient[row], ordersByClient[column], test_size=1/3, random_state=0)
     
    X_train=  pd.to_numeric(X_train.values).reshape(-1, 1)
    X_test =  pd.to_numeric(X_test.values).reshape(-1, 1)
 
    # Y_train = (Y_train).reshape(-1, 1)
    # Y_test = pd.to_numeric(Y_test).reshape(-1, 1)
    
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    
    # regresión lineal 
    # del set de entrenamiento y predicion de la misma
    y_train_pred= regressor.predict(X_train)
    y_test_pred= regressor.predict(X_test)
    # Podemos ver que el cliente tiene 300 pedidos
    return X_train, X_test, Y_train, Y_test, y_train_pred , y_test_pred


def runPrediction(db, x, y):  
    x_train, x_test, y_train,y_test = train_test_split(db[x],db[y],test_size=0.20, random_state=50)
    data = Data(x_train,x_test,y_train,y_test)
    data.x_train = data.x_train.sort_index()
    data.x_test = data.x_test.sort_index()
    data.y_train = data.y_train.sort_index()
    data.y_test = data.y_test.sort_index()
    
    """# **Custom Deep Learning Neural Network**"""
    def create_model():
      model = Sequential()
      model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
      model.add(Dense(64, kernel_initializer='normal', activation='relu'))
      model.add(Dense(1, kernel_initializer='normal'))
      model.compile(loss='mean_absolute_error', optimizer='adam')
      return model
    
    
    estimator_model = KerasRegressor(build_fn=create_model, verbose=1)
    #X_train = X_train.drop(columns='products')
    #X_test = X_test.drop(columns='products')
    history = estimator_model.fit(x_train, y_train, validation_split=0.2, epochs=500, batch_size=5000)
   
    return estimator_model, history, data
