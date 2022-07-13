# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:04:49 2022

@author: Luciano
"""

import tensorflow as tf
import helpers
import info
import regressionModel
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
from IPython.display import display
from lifetimes import  GammaGammaFitter
from lifetimes import BetaGeoFitter
from sklearn import metrics
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split 
from collections import Counter

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, cross_validate

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans
import xgboost as xgb
import time
import joblib

def calculateWeek(date):
    years= date.dt.strftime('%Y').astype(int) - 2009
    weeks= date.dt.strftime('%W').astype(int)
    return years * 51 + weeks

print(tf.__version__)

class Data(object):
    def __init__(self, x_train, x_test, y_train, y_test, y_test_pred = None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        
# Cargamos los datos procesados.

#df2.week = df2.week.astype(np.int16)
#df2.products = df2.products.astype(np.int16)
#df2.customerId = df2.customerId.astype(np.int16)
#df2.country = df2.country.astype(np.int16)
#df2.segmentation = df2.segmentation.astype(np.int16)
#df2.invoice = df2.invoice.astype(np.int32)
#df2.cluster = df2.cluster.astype(np.int16)
#df2.avgPrice = df2.avgPrice.astype(np.int16)
#df2.avgQuantity = df2.avgQuantity.astype(np.int16)
#df2.totalValue = df2.totalValue.astype(np.int16)
#joblib.dump(df2, "./data/df2.joblib")



def showPrections(data, y_pred):
    data.y_test_pred = y_pred

    info.modelLoss(data)

def compareAllModels():
       
    
    def runLinealRegression():
        resultRegression = regressionModel.runAllRegression(df2[['week', 'products', 'customerId', 'country', 'segmentation', 'invoice', 'cluster', 'totalValue']])
        print("**********  regression Lineal  *************")
        showPrections(resultRegression, resultRegression.y_test_pred)
        
        
    df2 = joblib.load("./data/df2.joblib")
    features = ['week', 'products', 'customerId', 'country', 'segmentation', 'invoice', 'cluster', 'avgPrice', 'avgQuantity']
    value = ['totalValue']
    #df2 = df2[df2.totalValue > 0]
    
    x_train, x_test, y_train,y_test = train_test_split(df2[features],df2[value],test_size=0.20, random_state=50)
    data= Data(x_train, x_test, y_train, y_test)
    
    
    # Procesamos los distintos modelos.
    
    # ******* LogisticRegression *******
    #https://es.wikipedia.org/wiki/Regresi%C3%B3n_log%C3%ADstica
    # logreg = LogisticRegression(solver='liblinear')
    # logreg.fit(x_train,y_train.totalValue.values)
    # joblib.dump(logreg, "./modelTrained/logreg.joblib")
    logreg = joblib.load("./modelTrained/logreg.joblib")
    y_pred=logreg.predict(x_test)
    print("**********  logreg  *************")
    showPrections(data, y_pred)
    
    # ******* GaussianNB *******
    #https://es.wikipedia.org/wiki/Clasificador_bayesiano_ingenuo
    #gau = GaussianNB()
    #gau.fit(x_train, y_train.totalValue.values)
    #joblib.dump(gau, "./modelTrained/gau.joblib")
    gau = joblib.load("./modelTrained/gau.joblib")
    y_pred2=gau.predict(x_test)
    print("**********  gau  *************")
    showPrections(data, y_pred2)

    
    # ******* randomForest  *******
    #https://es.wikipedia.org/wiki/Random_forest#Visualizaci%C3%B3n
    #randomForest = RandomForestClassifier()
    #randomForest.fit(x_train[0:15000],y_train[0:15000].totalValue.values)
    ##randomForest = joblib.dump(randomForest,"./modelTrained/randomForest.joblib")
    #data.y_test_pred = y_pred3
    # Guardamos los datos predecidos porque el modelo entrenado pesa mas de 50GB
    #y_pred3 = randomForest.predict(x_test)
    
    #joblib.dump(data,"./modelTrained/randomForestData.joblib")
    randomForestData = joblib.load('./modelTrained/randomForestData.joblib')
    print("**********  randomForest  *************")
    showPrections(randomForestData, randomForestData.y_test_pred)

    
    # ******* treeClassifier  *******
    # https://es.wikipedia.org/wiki/Aprendizaje_basado_en_%C3%A1rboles_de_decisi%C3%B3n
    #treeClassifier = DecisionTreeClassifier()
    #treeClassifier.fit(x_train, y_train.totalValue.values)
    #joblib.dump(treeClassifier,"./modelTrained/treeClassifier.joblib")
    treeClassifier = joblib.load("./modelTrained/treeClassifier.joblib")
    y_pred4 = treeClassifier.predict(x_test.drop(columns='invoice'))
    print("**********  treeClassifier  *************")
    showPrections(data, y_pred4)

    
    # ******* xgb  *******
    # https://medium.com/@jboscomendoza/tutorial-xgboost-en-python-53e48fc58f73
    #https://en.wikipedia.org/wiki/XGBoost
    #xgb= xgb.XGBClassifier(eval_metric='mlogloss')
    #xgb.fit(x_train, y_train.totalValue.values)
    #joblib.dump(xgb, "./modelTrained/xgb.joblib")
    xgb = joblib.load("./modelTrained/xgb.joblib")
    y_pred5= xgb.predict(x_test)
    print("**********  xgb  *************")
    showPrections(data, y_pred5)
        
    # ******* KNeighborsClassifier  *******
    # https://es.wikipedia.org/wiki/K_vecinos_m%C3%A1s_pr%C3%B3ximos
    #knei = KNeighborsClassifier()
    #knei.fit(x_train, y_train.totalValue.values)
    #joblib.dump(knei, "./modelTrained/knei.joblib")
    knei = joblib.load("./modelTrained/knei.joblib")
    y_pred6 = knei.predict(x_test)
    print("**********  knei  *************")
    showPrections(data, y_pred6)

    # ******* LinealRegression  *******
    #https://es.wikipedia.org/wiki/Regresi%C3%B3n_lineal
    runLinealRegression()
    
    # ******* TensorFlow  ******
    #https://www.adictosaltrabajo.com/2018/04/18/introduccion-a-machine-learning-con-tensorflow/

    
    







"""
Para un determinado cliente, vamos a aplicar un modelo de regresion lineal.
sobre la cantidad de items vs precio de la factura.
"""
#df2



def selectActivationFunction(df2,features, value):
    
    # Como leimos en el documento, las funciones de activación idonea para nuestro caso seria RELU (unidad lineal rectificada), ya que la 
    # mayoria como puede ser la tangente hiperbólica transforma en escala de entre -1 y 1.
    # Por lo tanto todos los modelos van a tener activation='relu'.
    
    epochs = 1000
    
    tinyData, tinyModelCreated, historyTiny = regressionModel.runPrediction(df2, features, value,epochs, "tiny")
    #*************** Métricas ***************
    #MAE 201.3477584566872
    #MSE 221064.49611529644
    #RMSE 470.1749633012124
    #R2 0.3819122877437404
    

    # prodriamos usar Leaky ReLU pero como los valores son todos positivos, los resultados serian muy parecidos a RELU
    # Igualmente pasamos a probarlo.
    
    tinyData, tinyModelCreated, historySoft = regressionModel.runPrediction(df2, features, value,epochs, "createTinySoftplusModel")
    #*************** Métricas ***************
    #MAE 214.12239379853906
    #MSE 235044.9109364556
    #RMSE 484.8143056227359
    #R2 0.3379548199110549
    
    # Efectivamente la mejor elección es RELU superando a Leaky Relu.
    


def selectInitializationMethod(df2,features, value):
    #Procederemos a hacer pruebas con distintos metódos de inicializacion que aprendimos con el documento.
    
    epochs = 1000
    
    tinyData, tinyModelCreated, historyZeros = regressionModel.runPrediction(df2, features, value,epochs, "createTinyZerosModel")

    #*************** Métricas ***************
    #MAE 427.36310673451834
    #MSE 534796.3914445966
    #RMSE 731.2977447282308
    #R2 0.0
    
    tinyData, tinyModelCreated, historyOnes = regressionModel.runPrediction(df2, features, value,epochs, "createTinyOnesModel")
    #*************** Métricas ***************
    #MAE 329.48626725711233
    #MSE 286333.67889214645
    #RMSE 535.1015594185336
    #R2 0.2441021666393649

    tinyData, tinyModelCreated, historyConstant = regressionModel.runPrediction(df2, features, value,epochs, "createTinyConstantModel")
    #*************** Métricas ***************
    #MAE 427.36311837343726
    #MSE 534796.4014298607
    #RMSE 731.297751555316
    #R2 0.0

    tinyData, tinyModelCreated, historyRandonN = regressionModel.runPrediction(df2, features, value,epochs, "createTinyRandomNormalModel")
    #*************** Métricas ***************
    #MAE 201.49204352385635
    #MSE 219181.8655691632
    #RMSE 468.16862941590097
    #R2 0.3825038443870139

    tinyData, tinyModelCreated, historyRandomU = regressionModel.runPrediction(df2, features, value,epochs, "createTinyRandomUniformModel")
    #*************** Métricas ***************
    #MAE 200.92401179437334
    #MSE 219914.90884895463
    #RMSE 468.95085973794164
    #R2 0.37961115372318444

    tinyData, tinyModelCreated, historyTruncate = regressionModel.runPrediction(df2, features, value,epochs, "createTinyTruncatedNormalModel")
    #*************** Métricas ***************
    #MAE 207.96662137294555
    #MSE 226752.6546914271
    #RMSE 476.1855254955017
    #R2 0.3584076871851699

    tinyData, tinyModelCreated, historyVariance = regressionModel.runPrediction(df2, features, value,epochs, "createTinyVarianceScalingModel")
    #*************** Métricas ***************
    #MAE 218.6613833500088
    #MSE 244747.858749476
    #RMSE 494.7199801397514
    #R2 0.32096483537367393

    tinyData, tinyModelCreated, historyLecunU = regressionModel.runPrediction(df2, features, value,epochs, "createTinyLecun_uniformModel")
    #*************** Métricas ***************
    #MAE 216.07452730682058
    #MSE 241649.38676778678
    #RMSE 491.5784645077394
    #R2 0.32745356689820293

    tinyData, tinyModelCreated, historyGlorotN = regressionModel.runPrediction(df2, features, value,epochs, "createTinyGlorot_normalModel")
    #*************** Métricas ***************
    #MAE 217.99889789907147
    #MSE 240830.93246998877
    #RMSE 490.74528267726504
    #R2 0.3223141507602595
    
    tinyData, tinyModelCreated, historyGlorotU = regressionModel.runPrediction(df2, features, value,epochs, "createTinyglorot_uniformModel")
    #*************** Métricas ***************
    #MAE 220.26618851546277
    #MSE 221366.63600545633
    #RMSE 470.4961593950118
    #R2 0.3715315584197708


    tinyData, tinyModelCreated, historyHe = regressionModel.runPrediction(df2, features, value,epochs, "createTinyHe_normalModel")
    #*************** Métricas ***************
    #MAE 233.98211219736794
    #MSE 243674.37422553418
    #RMSE 493.63384631276466
    #R2 0.30806816748899624
    
    
    # Analizando los resultados se encontraron 3 Soluciones que pueden ser adecuados para nuestro modelo
    # para hacer una elección vamos a volver a entrenar los modelos con el triple de epocas, que nos va a dar una mejor precision.
    epochs = 3000
        
    tinyData, tinyModelCreated, historyRandonN = regressionModel.runPrediction(df2, features, value,epochs, "createTinyRandomNormalModel")
    #*************** Métricas 3000 ***************
    #MAE 196.5088385755667
    #MSE 208559.64569424652
    #RMSE 456.6833100675418
    #R2 0.40918677066961817
    
    tinyData, tinyModelCreated, historyRandomU = regressionModel.runPrediction(df2, features, value,epochs, "createTinyRandomUniformModel")
    #*************** Métricas 3000***************
    #MAE 195.12832600580512
    #MSE 206166.3222523624
    #RMSE 454.0554176004978
    #R2 0.4198951466072828
    
    tinyData, tinyModelCreated, historyGlorotU = regressionModel.runPrediction(df2, features, value,epochs, "createTinyglorot_uniformModel")
    #*************** Métricas 3000 ***************
    #MAE 203.2901452182636
    #MSE 217156.65129092272
    #RMSE 466.00069880947893
    #R2 0.40558887123947585

    # Como se puede ver el metodo de random uniforme seria la mejor elección sobresaliento sobre los demás.

    
def selectActivationMethod(df2,features, value):

    epochs = 3000
    
    tinyData, tinyModelCreated, historySGD = regressionModel.runPrediction(df2, features, value,epochs, "createTinySGDModel")
    #*************** Métricas ***************
    #MAE 406.55348584791113
    #MSE 516837.20297979075
    #RMSE 718.9139051234096
    #R2 0.0
    
    tinyData, tinyModelCreated, historyRMSprop = regressionModel.runPrediction(df2, features, value,epochs, "createTinyRMSpropModel")
    #*************** Métricas ***************
    #MAE 221.64683179478254
    #MSE 249759.75859685134
    #RMSE 499.75970085317135
    #R2 0.3134222576961977
    
    tinyData, tinyModelCreated, historyAdadelta = regressionModel.runPrediction(df2, features, value,epochs, "createTinyAdadeltaModel")
    #*************** Métricas ***************
    #MAE 644.4604014008233
    #MSE 602810.536878396
    #RMSE 776.4087434324757
    #R2 -0.014191541952096154
    
    tinyData, tinyModelCreated, historyAdagrad = regressionModel.runPrediction(df2, features, value,epochs, "createTinyAdagradModel")
    #*******createTinyAdagradModel*******
    #*************** Métricas ***************
    #MAE 255.96510767635166
    #MSE 334597.44405917753
    #RMSE 578.4439852390009
    #R2 0.09584616016986125
    
    tinyData, tinyModelCreated, historyTiny = regressionModel.runPrediction(df2, features, value,epochs, "tiny") #ADAM
    #*************** Métricas ***************
    #MAE 201.3477584566872
    #MSE 221064.49611529644
    #RMSE 470.1749633012124
    #R2 0.3819122877437404
    

    tinyData, tinyModelCreated, historyNadam = regressionModel.runPrediction(df2, features, value,epochs, "createTinyNadamModel")
    #*************** Métricas ***************
    #MAE 223.0585544569714
    #MSE 225291.48567932015
    #RMSE 474.64880246274737
    #R2 0.3603267721607808
    
    tinyData, tinyModelCreated, historyFtrl = regressionModel.runPrediction(df2, features, value,epochs, "createTinyFtrlModel")
    #*******createTinyFtrlModel*******
    #*************** Métricas ***************
    #MAE 267.5457571312328
    #MSE 362972.7414281223
    #RMSE 602.4721914147758
        
        #R2 0.017215560662283247

    
    
    #Vemos que los que mejor resultado dan es el adam y el nadam, por lo tanto volvemos a probar pero ahora con el triple de epocas (3000)
    
    #*************** ADAM ***************
    #*************** Métricas ***************
    #MAE 196.2343183698976
    #MSE 209427.9964284901
    #RMSE 457.6330368630417
    #R2 0.4066528956461086
    
    
    
    #*******createTinyNadamModel*******
    #*************** Métricas ***************
    #MAE 198.17506201273238
    #MSE 213816.29022793597
    #RMSE 462.40273596502
    #R2 0.40405208255695146
    

  # Ambos nos dan un resultado similar, siendo el de Adam levemente superior en ambas pruebas por lo que vamos a elegirlo.
    
    
def selectSizeForModel(df2, features, value,epochs):
    epochs = 5000

    smallData, smallModelCreated, historySmall = regressionModel.runPrediction(df2, features, value,epochs, "small")
    #*************** Métricas ***************
    #MAE 195.16105076211707
    #MSE 209973.13983614164
    #RMSE 458.22826171695436
    #R2 0.41007228317942623
    
    mediumData, mediumModelCreated, historyMedium = regressionModel.runPrediction(df2, features, value,epochs, "medium")
    #*************** Métricas ***************
    #MAE 162.13696573463233
    #MSE 145510.05632445298
    #RMSE 381.4578041205252
    #R2 0.5957466886940599
     
    largeData, largeModelCreated, historyMedium = regressionModel.runPrediction(df2, features, value,epochs, "large")
    #*************** Métricas ***************
    #MAE 170.21900370294765
    #MSE 149453.05258203277
    #RMSE 386.59158369270375
    #R2 0.5777119156237287
    
    # Se prueba el mismo large con 7000 epocas.
    #*******large 7000*******
    #*************** Métricas ***************
    ##MAE 166.04953932378075
    #MSE 183061.99867507286
    #RMSE 427.8574513492465
    #R2 0.48382524068415456
    
    #largeData, largeModelCreated, historyMedium = regressionModel.runPrediction(df2, features, value,epochs, "extraLarge")
    #*************** Métricas ***************
    #MAE 175.36405999162523
    #MSE 146429.8042544013
    #RMSE 382.66147474550047
    #R2 0.5844303635353408
    
    # Guardamos el modelo elegido "mediano".
    import joblib
    joblib.dump(mediumData, "./modelTrained/medium/mediumData2.joblib")
    #joblib.dump(mediumModelCreated2, "./modelTrained/medium/mediumModelCreated.joblib") no se puede guardar.
    joblib.dump(historyMedium.history, "./modelTrained/medium/historyMedium2.joblib")
    
    
    # Ver gráfico de error del modelo elégido
    info.viewErrorForModel(mediumData)
    # Como se puede ver en el gráfico los errores de las predicciones se acumulan progresivamente mientras se va acercando al 0, en forma de 
    # campana.
    
    # como se puede visualizar todos los modelos funcionaron correctamente, logrando destacarse el de tamaño medio y el tamaño Extra largo.
    # Pero el problema es que con el de extra largo es  que el tiempo para entrenar es 15 veces mas alto obteniendo un resultado similar.
    # Lo que se puede visualiizar en todos los graficos de perdida de modelo que pasado los 3000 epocas, la perdida entre lo entrenado y de lo de testeo
    # empieza a aumentar, logrando asi una mejora en lo de entrenamiento cada vez mayor, pero el de testeo (que seria la informacion importante que nos permitira predecir
    #)en el futuro no logra ninguna mejora, por lo tanto hacer un modelo mas voluminoso como agregar etapas no mejoraria nuestro modelo.
    # Este ultimo dato tambien se puede visualizar viendo  la diferencia en un modelo largo 
    #entrenando 5000 épocas (con R2 de 0.57) y 7000 épocas (con R2 de 0.483)
        
    
def predictOneClient(df2, features, value):
    
    epochs =3000

    dataOneClient, modelCreatedOneClient, historyOneClient = regressionModel.runPrediction(df2[df2.customerId == 14911], features, value, epochs, "medium")
    #*************** Métricas ***************
    #MAE 241.29938197991788
    #MSE 149534.65438632236
    #RMSE 386.69710935863276
    #R2 0.3960890537408662
    # Se visualiza que un modelo entrenado con el find e predecir a un uníco cliente, el resultado es bastante inferior
    
    
def predictOneClientWithBestModel(db, modelCreated, history, features, value):
    epochs = 5000
    x_train, x_test, y_train,y_test = train_test_split(db[features],db[value],test_size=0.20)
    y_test_pred = modelCreated.predict(x_test)
    data= Data(x_train, x_test, y_train, y_test, y_test_pred)
    #print("*******" + size + "*******")
    info.modelLoss(data, history)
    
  #*************** Métricas ***************
  #MAE 184.23447757427508
  #MSE 111281.52418278359
  #RMSE 333.5888550038559
  #R2 0.6191192476747489
  # Se visualiza el comportamiento de nuestro modelo con los datos de todos los clientes es bastante superior R2 0.61 vs 0.39
  # Por lo tanto llegamos a la conclusion que los datos de un único cliente es bastante pobre para poder crear un modelo robusto.
    

    




