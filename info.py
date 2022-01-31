
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:23:52 2021

@author: Luciano
"""
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import helpers
import numpy as np
from sklearn import metrics
def initialInfo(dataSet):
    # [info] Verificamos si la base de datos fue leida y vemos un resumen del dataSet
    dataSet.info()
    
    # [info] Necesitamos comprobar de la base de datos, cada columna para ver el tipo de dato que contiene.
    display('************************* tipo de variables *************************', dataSet.dtypes)

    # [info] Averiguamos el periodo de tiempo de los pedidos.
    dateOrders = pd.to_datetime(dataSet['invoiceDate']).apply(lambda x:x.date())
    display('************************* Primeros pedidos del dataSet *************************',dataSet.head(5)) 
    display('************************* Primer pedido fue el :', dateOrders.min())
    display('************************* Ultimos pedidos del dataSet', dataSet.tail(5)) 
    display('************************* Ultimo pedido fue el :', dateOrders.max())
    display(' ************************* Lapso de periodo del dataSet es de', dateOrders.max() - dateOrders.min())
 
def ordersByCLient(groupByCustomer):
    plt.figure(figsize=(200, 20))
    groupByCustomer.plot('customerId',
                         'OrderQuantity',
                         title="Cantidad de pedidos por cliente",
                         ylabel="pedidos",
                         kind="scatter"
                         )
    
def viewBetterClients(better200):
    better200.plot('customerId',
                   'OrderQuantity',
                   title="Cantidad de pedidos por cliente - mejores 200",
                   ylabel="pedidos",
                   kind="scatter"
                   )
    # Imprimimos la cantidad de pedido min y max.
    display('Pedidos:','min',better200['OrderQuantity'].iloc[-1], 'max', better200['OrderQuantity'].iloc[0])
   
 
def viewOrdersByMonths(better200, dataSetByInvoice):
    for client in better200[0:10].customerId:
        ordersBymonth = helpers.getOrdersBymonth(client, dataSetByInvoice)
        # Usando la informacion obtenida de los primeros y últimos pedidos.
        firstMonth= pd.to_datetime('2009-01', format= '%Y-%M'  )
        lastMonth= pd.to_datetime('2012-12', format= '%Y-%M' )
        ordersBymonth.plot(kind='scatter',
                        x='month',
                        y='price',
                        figsize= (20,10),
                        sizes=(100,100),
                        xlim=(firstMonth,lastMonth),
                        fontsize = 20,
                        title=client
                        )
        
def viewResultRegression(resultRegression):
    X_train, X_test, Y_train, Y_test, y_train_pred , y_test_pred = resultRegression    
    
    # prediccion del set de test  
    plt.scatter(X_train, Y_train, color= 'blue')
    plt.plot(X_train, y_train_pred, color="green")
    plt.xlabel("items")
    plt.ylabel("priceInvoice")
    plt.title("train vs predict")
    plt.show()
    
    # prediccion del set de test  
    plt.scatter(X_test, Y_test, color='red')
    plt.plot(X_test, y_test_pred, color="violet")
    plt.xlabel("items")
    plt.ylabel("priceInvoice")
    plt.title("test vs predict")
    plt.show()
    
def viewResultRegressionByMonth(resultRegression, client):
    X_train, X_test, Y_train, Y_test, Y_train_pred , Y_test_pred = resultRegression 
    
    firstMonth= pd.to_datetime('2009-01', format= '%Y-%M'  ).toordinal()
    lastMonth= pd.to_datetime('2012-12', format= '%Y-%M' ).toordinal()
      
    # ROJO -> X e Y entrenamiento 
    plt.scatter(X_train, Y_train, color= 'red')
    
    # AZUL -> X linea predicción, uniendo para los dos sets
    plt.plot(X_train, Y_train_pred, color= 'blue')
    plt.plot(X_test, Y_test_pred, color= 'blue')
    
    # Verde -> X test, Y test 
    plt.scatter(X_test, Y_test, color="green")
       
    plt.xlabel("Mes")
    plt.ylabel("total Facturado")
    plt.xlim(firstMonth,lastMonth)
    plt.title("Cliente: " + str(client))
    plt.show()
    
def viewResultRegressionByOrder(resultRegression, labels, isDate):
    X_train, X_test, Y_train, Y_test, Y_train_pred , Y_test_pred = resultRegression 
    
    
    firstMonth= pd.to_datetime('2009-01', format= '%Y-%m'  ).toordinal()
    
    lastMonth= pd.to_datetime('2012-12', format= '%Y-%m' ).toordinal()
     
    plt.title(labels.title, fontsize=10)
    # ROJO -> X e Y entrenamiento 
    plt.scatter(X_train, Y_train, color= 'red')
    
    # AZUL -> X linea predicción, uniendo para los dos sets
    plt.plot(X_train, Y_train_pred, color= 'blue')
    plt.plot(X_test, Y_test_pred, color= 'blue')
    
    # Verde -> X test, Y test 
    plt.scatter(X_test, Y_test, color="green")
       
    plt.xlabel(labels.x,fontsize = 20)
    plt.ylabel(labels.y, fontsize = 20)
    if(isDate):
        print("pepe")
        plt.xlim(firstMonth,lastMonth)
            

    
    plt.show()
    
def modelLoss(history, data):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    #plt.xlim(00)
    #  plt.ylim(00,300)
    plt.legend(loc='upper right')
    #plt.savefig('plots/dnn_loss.png')
    plt.show()
    print("MAE" , metrics.mean_absolute_error(data.y_test, data.y_pred))
    ## https://es.wikipedia.org/wiki/Error_absoluto_medio
    print("MSE" , metrics.mean_squared_error(data.y_test, data.y_pred))
    ## https://es.wikipedia.org/wiki/Error_cuadr%C3%A1tico_medio
    print("RMSE" , np.sqrt(metrics.mean_squared_error(data.y_test, data.y_pred)))
    ## https://es.wikipedia.org/wiki/Ra%C3%ADz_del_error_cuadr%C3%A1tico_medio
    print("R2" , metrics.explained_variance_score(data.y_test, data.y_pred))
    
def show100Prediction(data):
        
    if len(data.y_test)> 100:
        q = 100
    else: 
        q = len(data.y_test)
        
    limY = foundYLimit(data)
        
    long = list(range(0,q))
    plt.figure(figsize=(200,25))
    plt.ylim(0,limY)
    #plt.ylim(1000)
    plt.plot(long, data.y_pred[0:q], label="prediction", linewidth=2.0,color='blue')
    plt.plot(long, data.y_test[0:q].totalValue,label="real_values", linewidth=2.0,color='red')
    #plt.savefig('plots/dnn_real_pred.png')
    plt.legend(loc="best")
    
    
def showAllPrediction(data):
        
    q = len(data.y_test)
    limY = foundYLimit(data)
    long = list(range(0,q))
    plt.figure(figsize=(200,25))
    plt.ylim(0,limY)
    #plt.ylim(1000)
    plt.plot(long, data.y_pred[0:q], label="prediction", linewidth=2.0,color='blue')
    plt.plot(long, data.y_test[0:q].totalValue,label="real_values", linewidth=2.0,color='red')
    #plt.savefig('plots/dnn_real_pred.png')
    plt.legend(loc="best")
    
    
def foundYLimit(data):
    maxYPred=data.y_pred.max()
    maxYTest= data.y_test.totalValue.values.max()
    return max(maxYPred, maxYTest)


def comparePredictionWithOrders(data):
    limit = foundYLimit(data)
    plt.figure()
    plt.scatter(data.y_test, data.y_pred)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, limit], [-100, limit])

def compare100PredictionWithOrders(data):
    limit = foundYLimit(data)
    plt.scatter(data.y_test[0:100], data.y_pred[0:100])
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, limit], [-100, limit])
    

