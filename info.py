
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
import seaborn as sns
from datetime import datetime
#import warnings

#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def initialInfo(dataSet):
    # [info] Necesitamos comprobar de la base de datos, cada columna para ver el tipo de dato que contiene.
    display('************************* tipo de variables *************************', dataSet.dtypes)

    # [info] Averiguamos el periodo de tiempo de los pedidos.
    dateOrders = pd.to_datetime(dataSet['invoiceDate']).apply(lambda x:x.date())
    display('************************* Primeros pedidos del dataSet *************************',dataSet.head(5)) 
    display('************************* Ultimos pedidos del dataSet', dataSet.tail(5)) 
    display('************************* Primer pedido fue el :', dateOrders.min())
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
    

    yMean = np.mean(np.concatenate((Y_test, y_test_pred)))

    plt.ylim(0, yMean * 5)
    # prediccion del set de test  
    plt.scatter(X_test, Y_test, color='red')
    plt.plot(X_test, y_test_pred, color="violet")
    plt.xlabel("Cantidad items")
    plt.ylabel("Total de la factura")
    plt.title("Test vs Prediction")
    plt.show()
    
    
    
def viewAllResultRegression(resultRegression):
     

    # prediccion del set de test  
    plt.scatter(resultRegression.x_test, resultRegression.y_test, color='red')
    plt.plot(resultRegression.x_test, resultRegression.y_test_pred, color="violet")
    plt.xlabel("Cantidad items")
    plt.ylabel("Total de la factura")
    plt.title("Test vs prediction")
    plt.show()
    
def viewResultRegressionByMonth(resultRegression, client):
    X_train, X_test, Y_train, Y_test, y_test_pred , y_train_pred = resultRegression 

    firstMonth= pd.to_datetime('2009-01', format= '%Y-%M'  ).toordinal()
    lastMonth= pd.to_datetime('2012-12', format= '%Y-%M' ).toordinal()
      
    # ROJO -> X e Y entrenamiento 
    plt.scatter(X_train, Y_train, color= 'red')
    
    # AZUL -> X linea predicción, uniendo para los dos sets
    plt.plot(X_train, y_train_pred, color= 'blue')
    plt.plot(X_test, y_test_pred, color= 'blue')
    
    # Verde -> X test, Y test 
    plt.scatter(X_test, Y_test, color="green")
       
    plt.xlabel("Mes")
    plt.ylabel("total Facturado")
    plt.xlim(firstMonth,lastMonth)
    plt.title("Cliente: " + str(client))
    plt.show()
    
def viewResultRegressionByOrder(result, labels, isDate):
    yMean = np.mean(np.concatenate((result.y_test, result.y_train)))
    yStd = np.std(np.concatenate((result.y_test, result.y_train)))

       
    plt.figure(figsize=(50,20))
    
    plt.ylim(0, yMean + (5 * yStd))
    plt.title(labels.title, fontsize=15)
    if(isDate):
        firstMonth= pd.to_datetime('2009-01', format= '%Y-%m').toordinal()
        
        lastMonth= pd.to_datetime('2012-12', format= '%Y-%m').toordinal()
        print("pepe")
        plt.xlim(firstMonth,lastMonth)
                       
    else:
        print("NO pepe")
        xMean = np.mean(np.concatenate((result.x_test, result.x_train)))
        xStd = np.std(np.concatenate((result.x_test, result.x_train)))
        plt.xlim(0, xMean + (5 * xStd))

    # ROJO -> X e Y entrenamiento 
  
    plt.scatter(result.x_train, result.y_train, color= 'red')
   
    
    # AZUL -> X linea predicción
    plt.plot(result.x_test, result.y_test_pred, color= 'blue')
    
    # Verde -> X test, Y test 
    plt.scatter(result.x_test, result.y_test, color="green")
       
    plt.xlabel(labels.x,fontsize = 20)
    plt.ylabel(labels.y, fontsize = 20)
 
    
    
    plt.show()
    plt.figure()
    #plt.title("Regresión lineal")
    #plt.xlabel(labels.x,fontsize = 20)
    #plt.ylabel(labels.y, fontsize = 20)
    #plt.plot(result.x_test, result.y_test_pred, color= 'blue')
    
    
def modelLoss(data, history= None):    
    if history: 
        plt.figure(figsize=(20,10))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.xlim(00)
      
        #plt.ylim(00,300)
        plt.legend(loc='upper right')
    #plt.savefig('plots/dnn_loss.png')
        plt.show()
    print("*************** Métricas ***************")
    print("MAE" , metrics.mean_absolute_error(data.y_test, data.y_test_pred))
    ## https://es.wikipedia.org/wiki/Error_absoluto_medio
    print("MSE" , metrics.mean_squared_error(data.y_test, data.y_test_pred))
    ## https://es.wikipedia.org/wiki/Error_cuadr%C3%A1tico_medio
    print("RMSE" , np.sqrt(metrics.mean_squared_error(data.y_test, data.y_test_pred)))
    ## https://es.wikipedia.org/wiki/Ra%C3%ADz_del_error_cuadr%C3%A1tico_medio
    print("R2" , metrics.explained_variance_score(data.y_test, data.y_test_pred))
    
def show100Prediction(data):
        
    if len(data.y_test)> 100:
        q = 100
    else: 
        q = len(data.y_test)
        
    limY = foundYLimit(data,100)
        
    long = list(range(0,q))
    plt.figure(figsize=(200,25))
    plt.ylim(0,limY)
    #plt.ylim(1000)
    plt.plot(long, data.y_test_pred[0:q], label="prediction", linewidth=2.0,color='blue')
    plt.plot(long, data.y_test[0:q],label="real_values", linewidth=2.0,color='red')
    #plt.savefig('plots/dnn_real_pred.png')
    plt.legend(loc="best")
    
    
def showAllPrediction(data):
        
    q = len(data.y_test)
    limY = foundYLimit(data)
    long = list(range(0,q))
    plt.figure(figsize=(200,25))
    plt.ylim(0,limY)
    #plt.ylim(1000)
    plt.plot(long, data.y_test_pred[0:q], label="prediction", linewidth=2.0,color='blue')
    plt.plot(long, data.y_test[0:q],label="real_values", linewidth=2.0,color='red')
    #plt.savefig('plots/dnn_real_pred.png')
    plt.legend(loc="best")
    
    
def foundYLimit(data, q= None):
    if (q):  
        maxYPred=data.y_test_pred[0:q].max()
        maxYTest= data.y_test[0:q].max()
    else:
        maxYPred=data.y_test_pred.max()
        maxYTest= data.y_test.max()
    return max(maxYPred, maxYTest.values)


def comparePredictionWithOrders(data):
    limit = foundYLimit(data)
    plt.figure()
    plt.scatter(data.y_test, data.y_test_pred)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, limit], [-100, limit])

def compare100PredictionWithOrders(data):
    limit = foundYLimit(data)
    plt.scatter(data.y_test[0:100], data.y_test_pred[0:100])
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, limit], [-100, limit])
    
def showXY(x,y):
    q = len(y)
    limY = max(y)
    limX = max(x)
    long = list(range(0,q))
    plt.figure(figsize=(25,10))
    #plt.ylim(0,limY)
    #plt.xlim(0,limX)
    plt.plot(y[0:q], x[0:q], label="prediction", linewidth=2.0,color='blue')
    plt.scatter(y[0:q], x[0:q], label="prediction", linewidth=2.0,color='green')
    #plt.plot(long, y[0:q],label="real_values", linewidth=2.0,color='red')
    #plt.savefig('plots/dnn_real_pred.png')
    plt.legend(loc="best")
    

def analysisRMF(rmf):
    plt.figure()
    plt.subplot(3, 1, 1)
    sns.distplot(rmf['recency'])
    plt.subplot(3, 1, 2)
    sns.distplot(rmf['monetary'])
    plt.subplot(3, 1, 3)
    sns.distplot(rmf['frequency'])


    # Aplicamos una transformación logaritmica.
    rmfData = rmf[['recency','frequency','monetary']]
    rmfDataLog =  np.log(rmfData)
 
    plt.figure()
    # Vemos el resultado y comparamos
    plt.subplot(3, 1, 1, title="transformación Logaritma")
    sns.distplot(rmfDataLog['recency']).set_ylabel("recency")
    plt.subplot(3, 1, 2)
    sns.distplot(rmfDataLog['monetary']).set_ylabel("monetary")
    plt.subplot(3, 1, 3)
    sns.distplot(rmfDataLog['frequency']).set_ylabel("frequency")
    
def showElbowMethod(rmfDataLog):
    # Con el metdo del codo tratatermos de interpetrar los numeros de cluster
    from sklearn.cluster import KMeans

    distortions = []
    # postulamos clusters entre 2 y 10.
    K = range(2,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(rmfDataLog)
        distortions.append(kmeanModel.inertia_)
        
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distorción')
    plt.title('Metodo del codo mostrando el Nro de K')
    plt.show()

def showClusterImportance(rmf):
    
    rmfDataMean = rmf[['recency','frequency','monetary','cluster']].mean()
    clusterGrouped = rmf[['recency','frequency','monetary','cluster']].groupby(['cluster']).mean()
    
    # calculamos la importancia relativa en cada grupo
    
    relative_imp = (clusterGrouped / rmfDataMean) - 1
    relative_imp.round(2)


    relative_imp = relative_imp.drop(['cluster'], axis=1)
    # Plot heatmap

    plt.figure(figsize=(20, 20))
    plt.title('Importancia de los atributos')
    sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
    sns.set(font_scale=3)
    ## TODO hacer análisis de este gráfico en el documento
    plt.show()

def showSegmentation(rfm):
    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return "{:.1f}%\n({:d})".format(pct, absolute)
    
    clientsBySegments = rfm.groupby(rfm.segment).agg({'monetary': 'count'} )
    clients = clientsBySegments.monetary
    segments = clients.index    
    fig, ax = plt.subplots(figsize=(35, 35))
   
    wedges, texts, autotexts = ax.pie(clients, autopct=lambda pct: func(pct, clients),
                                      textprops=dict(color="black"),pctdistance=0.7, radius= 1.2, 
                                      colors=['#00bb2d','#fe4400','#8673a1', 
                                              '#64ff00','#ff0066','#f7ff00',
                                              '#d7b100','#881100','#0083ff',
                                              '#8fed00','#f80000'])
    
    
    ax.legend(wedges, segments,
              title="Segmentos",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=30)
    plt.setp(autotexts, size=35, weight="bold")
    plt.rcParams['font.size'] = 100
    ax.set_title("Distribucion de clientes en cada segmentación")
    plt.show()

def viewErrorForModel(data):
        
    error = data.y_test_pred - data.y_test.totalValue.values
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('default')
    plt.hist(error, bins=200)
    plt.xlabel("Prediction Error [MPG]")
    plt.xlim(-3000,3000)
    _ = plt.ylabel("Count")