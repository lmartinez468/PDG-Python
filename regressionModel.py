# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:59:25 2021

@author: Luciano
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import info
from datetime import datetime 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import random

class Data(object):
    def __init__(self, x_train, x_test, y_train, y_test, y_test_pred = None, y_train_pred = None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        self.y_train_pred = y_train_pred



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
    
    return X_train, X_test, Y_train, Y_test, y_test_pred , y_train_pred 

def runRegressionByOrder(ordersByClient, row, column):
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
    result = Data(X_train, X_test, Y_train, Y_test, y_test_pred, y_train_pred )
    return result


def runPrediction(db, x, y, epochs, size):  
    x_train, x_test, y_train,y_test = train_test_split(db[x],db[y],test_size=0.20, random_state=50)
    data = Data(x_train,x_test,y_train,y_test)
    data.x_train = data.x_train.sort_index()
    data.x_test = data.x_test.sort_index()
    data.y_train = data.y_train.sort_index()
    data.y_test = data.y_test.sort_index()
    
    
    

    estimator_model = getModel(size)
    
    

    #X_train = X_train.drop(columns='products')
    #X_test = X_test.drop(columns='products')
    # ,  verbose=0
    history = estimator_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=5000)
    data.y_test_pred = estimator_model.predict(data.x_test)
    print("*******" + size + "*******")
    info.modelLoss(data, history)
    info.show100Prediction(data)
    info.comparePredictionWithOrders(data)
    info.compare100PredictionWithOrders(data)
    return data, estimator_model, history



def createTinyModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='RandomUniform',activation='relu'))
  model.add(Dense(1, kernel_initializer='RandomUniform'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinySGDModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='normal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='SGD')
  
  return model

def createTinyRMSpropModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='normal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='RMSprop')
  
  return model

def createTinyAdadeltaModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='normal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='Adadelta')
  
  return model

def createTinyAdagradModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='normal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='Adagrad')
  
  return model

def createTinyAdamaxModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='normal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='Adamax')
  
  return 

def createTinyNadamModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='normal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='Nadam')
  
  return model

def createTinyFtrlModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='normal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='Ftrl')
  
  return model

def createTinyZerosModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='zeros',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyOnesModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='ones',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyConstantModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='constant',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyRandomNormalModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='RandomNormal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyRandomUniformModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='RandomUniform',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyTruncatedNormalModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='TruncatedNormal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyVarianceScalingModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='VarianceScaling',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyLecun_uniformModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='LecunUniform',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyGlorot_normalModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='GlorotNormal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyglorot_uniformModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='GlorotUniform',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createTinyHe_normalModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='HeNormal',activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model



def createTinySoftplusModel():
 
  model = Sequential()
  #model = Model()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='RandomUniform',activation='softplus'))
  model.add(Dense(1, kernel_initializer='RandomUniform'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  return model

def createSmallModel():
 
  model = Sequential()
  #model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, input_dim=9, kernel_initializer='normal',activation='relu'))
  model.add(Dense(16, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(1, kernel_initializer='RandomUniform'))
  model.compile(loss='mean_absolute_error', optimizer='Adam')
  return model

def createMediumModel():
  
  model = Sequential()

  model.add(Dense(64, input_dim=9, kernel_initializer='RandomUniform',activation='relu'))
  model.add(Dense(64, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(64, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(1, kernel_initializer='RandomUniform'))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  return model

def createLargeModel():
 
  model = Sequential()

  model.add(Dense(512, input_dim=9, kernel_initializer='RandomUniform',activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))

  model.add(Dense(1, kernel_initializer='RandomUniform'))
  
  model.compile(loss='mean_absolute_error', optimizer='adam')
  return model


def createLargeModel():
 
  model = Sequential()

  model.add(Dense(512, input_dim=9, kernel_initializer='RandomUniform',activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))

  model.add(Dense(1, kernel_initializer='RandomUniform'))
  
  model.compile(loss='mean_absolute_error', optimizer='adam')
  return model


def createExtraLargeModel():
 
  model = Sequential()

  model.add(Dense(512, input_dim=9, kernel_initializer='RandomUniform',activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))
  model.add(Dense(512, kernel_initializer='RandomUniform', activation='relu'))

  model.add(Dense(1, kernel_initializer='RandomUniform'))
  
  model.compile(loss='mean_absolute_error', optimizer='adam')
  return model


def getModel(size):

    switcherModel = {
        "createTinySoftplusModel":KerasRegressor(build_fn=createTinySoftplusModel, verbose=1),
        "createTinyZerosModel":KerasRegressor(build_fn=createTinyZerosModel, verbose=1),
        "createTinyOnesModel":KerasRegressor(build_fn=createTinyOnesModel, verbose=1),
        "createTinyConstantModel":KerasRegressor(build_fn=createTinyConstantModel, verbose=1),
        "createTinyRandomNormalModel":KerasRegressor(build_fn=createTinyRandomNormalModel, verbose=1),
        "createTinyRandomUniformModel":KerasRegressor(build_fn=createTinyRandomUniformModel, verbose=1),
        "createTinyTruncatedNormalModel":KerasRegressor(build_fn=createTinyTruncatedNormalModel, verbose=1),
        "createTinyVarianceScalingModel":KerasRegressor(build_fn=createTinyVarianceScalingModel, verbose=1),
        "createTinyLecun_uniformModel":KerasRegressor(build_fn=createTinyLecun_uniformModel, verbose=1),
        "createTinyGlorot_normalModel":KerasRegressor(build_fn=createTinyGlorot_normalModel, verbose=1),
        "createTinyglorot_uniformModel": KerasRegressor(build_fn=createTinyglorot_uniformModel, verbose=1),
        "createTinyHe_normalModel": KerasRegressor(build_fn=createTinyHe_normalModel, verbose=1),
        "createTinySGDModel": KerasRegressor(build_fn=createTinySGDModel, verbose=1),
        "createTinyRMSpropModel": KerasRegressor(build_fn=createTinyRMSpropModel, verbose=1),
        "createTinyAdadeltaModel": KerasRegressor(build_fn=createTinyAdadeltaModel, verbose=1),
        "createTinyAdagradModel": KerasRegressor(build_fn=createTinyAdagradModel, verbose=1),
        "createTinyAdamaxModel": KerasRegressor(build_fn=createTinyAdamaxModel, verbose=1),
        "createTinyNadamModel": KerasRegressor(build_fn=createTinyNadamModel, verbose=1),
        "createTinyFtrlModel": KerasRegressor(build_fn=createTinyFtrlModel, verbose=1),
        "tiny": KerasRegressor(build_fn=createTinyModel, verbose=1),
        "small": KerasRegressor(build_fn=createSmallModel, verbose=1),
        "medium": KerasRegressor(build_fn=createMediumModel, verbose=1),
        "large": KerasRegressor(build_fn=createLargeModel, verbose=1),
        "large": KerasRegressor(build_fn=createExtraLargeModel, verbose=1),        
    }
    return switcherModel.get(size)
def runAllRegression(dataSet):
    """
    Dado un determinado usuario, se procedera a dividir sus pedidos
    2/3 sera utilizado para entrenar el modelo (X_train, y_train)
    1/3 sera utilizado para predecir.(X_test, Y_test)
    retornando estos valores mas la predicción de 
    tanto del set de entrenamiento como el set de testeo.
    """
    yy= dataSet['totalValue']
    X_train, X_test, Y_train, Y_test = train_test_split(dataSet.drop(columns=["totalValue"]), yy.values, test_size=1/3, random_state=0)

    #X_train = pd.to_numeric(X_train).reshape(-1, 1)
    #X_test = pd.to_numeric(X_test).reshape(-1, 1)
    Y_train = pd.to_numeric(Y_train).reshape(-1, 1)
    Y_test = pd.to_numeric(Y_test).reshape(-1, 1)

    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    
    # regresión lineal 
    # del set de entrenamiento y predicion de la misma
    #y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    return  Data(X_train, X_test, Y_train, Y_test, y_test_pred)

def useModelCreated(data, modelCreated):
    data.y_test_pred = modelCreated.predict(data.x_test)
    info.show100Prediction(data)
    info.showAllPrediction(data)
    info.comparePredictionWithOrders(data)