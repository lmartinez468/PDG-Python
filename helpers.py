# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:29:19 2021

@author: Luciano
"""
from datetime import datetime, timedelta
import pandas as pd
from IPython.display import display
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import regressionModel
import info
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
import constants
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#warnings.simplefilter('default')

def loadDataSet():
    url = "./dataSet.csv"
    column_names = ['id',
                    'invoice',
                    'itemId',
                    'description',
                    'quantity',
                    'invoiceDate',
                    'price',
                    'customerId',
                    "country",
                    "null",
                    "OrderQuantity"
                    ]

    return pd.read_csv(
        url,
        names = column_names,
        skipinitialspace = True,
        low_memory = False
    )

def filterDataSet(dataSet):
    # configuracion para poder ver todas las columnas al imprimir
    pd.set_option('display.max_columns', None)
    
    # Vemos que invoiceDate esta como Object, necesitaremos cambiar el tipo de dato a dateTime.
    dataSet['invoiceDate'] = pd.to_datetime(dataSet['invoiceDate'])
    
    # Vemos que invoiceDate esta como Object, necesitaremos cambiar el tipo de dato a dateTime.
    dataSet['customerId'] = dataSet['customerId'].astype('int32')
    
    # Buscaremos si existen devoluciones de pedidos, esto lo podemos visualizar viendo las cantidades de los invoices, las cuales sean menores a 0 
    negativeQuantity = dataSet.quantity < 0
    dataSet[negativeQuantity]
    display('Devoluciones de pedidos',dataSet[negativeQuantity])
    
    # Elimina aquellos campos vacio en el id de los clientes.
    dataSet = dataSet[dataSet['customerId'].notna()]
    
    # Elimina aquellos pedidos con productos mal cargados con precios igual a 0 o negativos.
    dataSet = dataSet[dataSet['price'] > 0]
    dataSet = dataSet[dataSet['quantity'] > 0]
    dataSet.reset_index(drop=True,inplace=True)
    
    # rmfData = rmfData.dropna() # agregar en el helper de los filtros
    # rmfData = rmfData[rmfData.monetary > 1] # agregar en el helper de los filtos
    
    # Como encontramos devoluciones procederemos a eliminarlas del dataSet (TODO ¿ elimnar la factura asociada ?)
    return dataSet[~negativeQuantity]

    
    
def groupByInvoice(dataSet):
     dataSet['variety'] = 1
     dataSet['price'] = dataSet['price'] * dataSet['quantity']
     dataSet = dataSet.groupby(dataSet['invoice'], as_index = False).agg({
         "invoiceDate": "first",
         "price": 'sum',
         'customerId': 'first',
         'variety': 'count',
         'quantity': 'sum'
         })
     dataSet = dataSet.groupby('invoiceDate').agg({'price': 'sum','quantity': "sum",'variety': 'sum'})
     dataSet['invoiceDate']= dataSet.index
     dataSet['invoiceDate'] = dataSet['invoiceDate'].apply(datetime.toordinal)
     return dataSet


def getBestProductsLastMonth(dataSet):
    bestProductLastMonth= dataSet[dataSet.invoiceDate > datetime(2011,11,1)]
    bestProductLastMonth= bestProductLastMonth.groupby(bestProductLastMonth.itemId.tolist(), as_index=False, dropna=False).size().rename(columns={'index':'itemId', 'size': 'quantity'})
    bestProductLastMonth = bestProductLastMonth.sort_values('quantity', ascending=False)[0:20]
    bestProductLastMonth.head(10)
    print( bestProductLastMonth.head(10))
    bestProductLastMonth.to_csv('bestProductLastMonth.csv')
    
    
def groupByCustomer(groupByInvoice):
    
    # A la columna de OrderQuantity le asigneramos el valor de 1, cantidad de ordenes por pedido, que luego usaremos.
    groupByInvoice['OrderQuantity'] = 1
    
    # Agrupamos por customerId.
    groupByCustomer = groupByInvoice.groupby(groupByInvoice['customerId'], as_index = False).agg({
        "invoiceDate": "first",
        "price": 'sum',
        'OrderQuantity' : 'count',
        "variety": 'sum',
        })
    

    
    # Ordenaremos de mayor a menor a los clientes por cantidad de pedidos.
    return groupByCustomer.sort_values('OrderQuantity', ascending=False)

def getOrdersBymonth(client, dataSetByInvoice):
    ordersByClient = dataSetByInvoice.customerId == pd.to_numeric(client)
    # Se agrupa todos los pedidos de un cliente por mes.
    ordersBymonth=dataSetByInvoice[ordersByClient].groupby(dataSetByInvoice['invoiceDate'].dt.strftime('%Y-%m'), as_index = True)['price'].sum().sort_index(ascending = True)
    # convertimos la serie en un dataFrame.
    ordersBymonth = pd.DataFrame(data=ordersBymonth)
    # asignamos una nueva columna para los meses.
    ordersBymonth['month']= pd.to_datetime(ordersBymonth.index)
    return ordersBymonth


def calculateWeek(date):
    years= date.dt.strftime('%Y').astype(int) - 2009
    weeks= date.dt.strftime('%W').asvbgfdtype(int)
    return years * 51 + weeks


def exportResultPrediction(data):
    dataToExport = data.x_test
    dataToExport["totalValue"] = data.y_test
    dataToExport["totalValuePredicted"] = data.y_test_pred
  
    dataToExport = dataToExport.groupby('customerId').agg({
    'invoice': 'count',
    'customerId': 'first',
    'country': 'first',
    'segmentation': 'first',
    'cluster': 'first',
    'avgPrice': 'mean',
    'avgQuantity': 'mean',
    'totalValuePredicted': 'first',
    'totalValue': 'first',
    'products': "first"
    })
    
    dataToExport = dataToExport.rename(columns={'totalValue':'nextOrder', 'totalValuePredicted': 'nextOrderPredicted'})
    
    ## Aplicamos distintos filtros     
    dataToExport = dataToExport[dataToExport.cluster != 404]    
    dataToExport = dataToExport[dataToExport.invoice <=1]
    dataToExport = dataToExport[dataToExport.nextOrderPredicted > 1]        
    return dataToExport

## Utilizamos  los Códigos de los paises obtenidos de la ISO
def loadCountries():
    url = "./country-codes.csv"
    column_names = ['name',
                    'null',
                    'null2',
                    'countryCode',
                    'null3'
                    ]
    data = pd.read_csv(
            url,
            names = column_names,
            skipinitialspace = True,
            low_memory = False,
            header=0,        
                )
    listCountries = data[['name', 'countryCode']]
    
## Necesitamos renombrar 1 Sólo pais para que coincida con la base de datos
    listCountries['name']= listCountries.name.replace(to_replace ="United States",
                     value ="USA")
    return listCountries

def loadFakesNames(data):
    fakesNames= loadNames()
    data['name']= fakesNames.name.values[0:len(data)]
    return data

def loadNames():
    url = "./names.csv"
    column_names = ['name'
                    ]
    data = pd.read_csv(
            url,
            names = column_names,
            skipinitialspace = True,
            low_memory = False,
            header=0,        
                )
    return data

def addBestProducts(dataSet,dataToExport):
    bestProductByClient = dataSet.groupby('customerId')['itemId'].apply(list).apply(lambda x : Counter(x).most_common(5))
   
    dataToExport['bestProduct'] = dataToExport.customerId.apply(lambda client: bestProduct(client,bestProductByClient ))
   
    return dataToExport
    
def bestProduct(client,products):
    match = products.loc[products.index == client]
    if match.size > 0:
        return match.values[0]
    else:
        return -404
    
def bestAllProducts(data):
    bestProduct = data.groupby(data.itemId.tolist(), as_index=False, dropna=False).size().rename(columns={'index':'itemId', 'size': 'quantity'})
    ## ordenamos y guardamos los primeros 20
    return bestProduct.sort_values('quantity', ascending=False)[0:20]
  

def exportItemsDetails(data):
    allItems = data.groupby('itemId').agg({'description':'first', "price": "mean", "quantity": "mean"})
    allItems['quantity'] = allItems['quantity'].round() 
    print(allItems)
    allItems.to_csv('allItems.csv')
    
    
def bestProduct2(item,bestProductLastMonth):
    match = bestProductLastMonth.loc[item.itemId == bestProductLastMonth.itemId]
    if match.size > 0:
        return match.values[0]
    else:
        return -404

def calculateWeek(date):
    years= date.dt.strftime('%Y').astype(int) - 2009
    weeks= date.dt.strftime('%W').astype(int)
    return years * 51 + weeks

class Data(object):
    def __init__(self, x_train, x_test, y_train, y_test, y_test_pred = None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        
def cohortAnalysis(dataSet):
        
    # primeros sobre la base de datos, tenemos que averiguar para cada cliente, la fecha del primer pedido que hizo, que lo agregaremos
    # a una nueva columna llamada 'firstOrder'

    customerGrouped = dataSet.groupby('customerId')['invoiceDate']

    ## aplicaremos a cada fila del dataset el pedido minimo de cada cliente.
    cohort = dataSet ## creando una nueva dataframe, para no utilizar el dataset.

    ## creamos columnas para el año y mes del pedido, tambien así para el primer pedido de cada cliente.
    cohort['monthOrder'] = cohort['invoiceDate'].apply(lambda x: x.month)
    cohort['yearOrder'] = cohort['invoiceDate'].apply(lambda x: x.year)

    cohort['firstOrder'] = customerGrouped.transform('min')
    cohort['firstOrder'] = cohort['firstOrder'].apply(lambda x: datetime(x.year, x.month, 1))
    cohort['firstMonthOrder'] = cohort['firstOrder'].apply(lambda x: x.month)
    cohort['firstYearOrder'] = cohort['firstOrder'].apply(lambda x: x.year)

    ## procedemos a agregar otra nueva columna llamada 'diff', que seria la diferencia de dias entre el primer pedido y el pedido.
    cohort['diff'] = (cohort['yearOrder'] - cohort['firstYearOrder']) * 12 + cohort['monthOrder']- cohort['firstMonthOrder']


    ## agrupamos la base de datos de agrupados por la primer orden y los meses de diferencia.
    grouping = cohort.groupby(['firstOrder', 'diff'])
    cohortToPlot = grouping['customerId'].apply(pd.Series.nunique)

    cohortToPlot = cohortToPlot.reset_index()
    cohortToPlot = cohortToPlot.pivot(index='firstOrder',columns='diff',values='customerId')

    ## obtengo las cantidad de usuarios que compraron en el mes 0, que nos marca el tamaño
    cohort_sizes = cohortToPlot.iloc[:,0]

    # 
    cohortToPlot = cohortToPlot.divide(cohort_sizes, axis=0)

    # Redondeo los datos a tres digitos y multiplicamos por 100 para obtener el resultado porcentual.
    cohortToPlot =cohortToPlot.round(3)


    # Graficamos
    plt.style.use('default')
    plt.figure(figsize=(30, 8))
    plt.title('Ratio de usuarios que vuelven a comprar')
    sns.heatmap(data = cohortToPlot,annot = True,fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap = 'BuGn')
    plt.show()

def calculateRMF(dataSet):
    
    ## verificamos la fecha del ultimo pedido que tomariamos como fin del dataSet
    # se agrega +1, para utilizar valores igual o mayores a 0, ya que 0 luego no se puede utilizar el logartimo
    endDate = dataSet['invoiceDate'].max() + timedelta(1)
    
    ## Creamos una nueva columna con la multiplicacion del precio del producto por la cantidad.
    dataSet['totalValue']= dataSet['quantity'] * dataSet['price']
    
    ## Creamos una nueva variable agrupando el dataSet por cliente y creando las 3 columnas que nos interesan para este analisis que son las 
    ## frecuencia, recencia y valor monetario.
    
    
    ## Removemos los valores negativos o igual a 0.
    
    
    ## fixear con esto
 
    rmf = dataSet.groupby(['invoice']).agg({'invoiceDate': 'first',
                                                                'invoice': 'first',
                                                                 'totalValue': 'sum',
                                                                 'customerId': 'first',
                                                                 'quantity': 'sum'})
    rmf['var']= rmf['totalValue']
    rmf['media']= rmf['totalValue']
    rmf = rmf.groupby(['customerId']).agg({'invoiceDate': lambda x: (endDate - x.max()).days,
                                                                'invoice': 'count',
                                                                 'totalValue': 'sum',
                                                                 'var': 'var',
                                                                 'media': 'mean',
                                                                 'quantity': 'sum'
                                                                 }).rename(columns={'invoiceDate': 'recency',
                                                                                    'invoice': 'frequency','totalValue': 'monetary'})   
    rmf = rmf[rmf.monetary > 1]
    rmf['std']= rmf['var'].apply(lambda x: np.sqrt(x))
                                                                      

    
    rmf['monetary'] = rmf['monetary'].apply(lambda x: int(round(x))) 
    
    ## para cada uno de estas 3 nuevas columnas, vamos a segmentarlas

    ## Para poder saber el grado de segmentacion 

    # Crearemos una variable donde calcularemos los cuartiles de cada columna 
    # ascendingLabels donde un cliente con valor numerico mas bajo es favorable, ya que la última transaccion fue reciente, por lo que invertiremos el start y stop.
    # ejemplo valores de recencia de [10,30,20,40] -> corresponde a cuantiles [4,3,2,1]
    invertLabels = range(4, 0, -1)
    recencyQuartiles = pd.qcut(rmf['recency'], 4, labels = invertLabels)

    # para la frecuencia y el valor monetario, no se invertira el start y stop, ya que mientras mas alto sea el valor númerico es mas favorable
    normalLabels = range(1,5)

    # Como dos cuantiles son igual a 1, necesitamos remover los duplicados, por lo tanto se selecciona 5 y
    # se agrega como agrumento que elimina "drop" los duplicados.
    frequencyQuartiles = pd.qcut(rmf['frequency'], 5, labels = normalLabels, duplicates= "drop")
    monetaryQuartiles = pd.qcut(rmf['monetary'], 4, labels = normalLabels)

    ## Agregamos a la tabla las tres nuevas columnas calculadas
    rmf = rmf.assign(R = recencyQuartiles.values,
                         F = frequencyQuartiles.values,
                         M = monetaryQuartiles.values,
                         )
    ## Agregamos una nueva columna concatenando RMF, donde el intervalo sera de 111 a 444, donde mientras mayor sea el valor, el cliente es mas importante
    rmf = rmf.assign(RMF = rmf['R'].astype(str) + rmf['F'].astype(str) + rmf['M'].astype(str))
    rmf["rfScore"] = (rmf['R'].astype(str) + rmf['F'].astype(str))

    # analysisRFM

    plt.figure(figsize=(15,10))
    plt.subplot(3, 1, 1)
    sns.distplot(rmf['recency']).set_xlabel("recency")
    plt.subplot(3, 1, 2)
    sns.distplot(rmf['monetary']).set_xlabel("monetary")
    plt.subplot(3, 1, 3)
    sns.distplot(rmf['frequency']).set_xlabel("frequency")


    # Aplicamos una transformación logaritmica.
    rmfData = rmf[['recency','frequency','monetary']]
    #rmfData = rmfData.dropna() # agregar en el helper de los filtros
    #rmfData = rmfData[rmfData.monetary > 1] # agregar en el helper de los filtos
    rmfDataLog =  np.log(rmfData)
    ##rmfDataLog = rmfDataLog.dropna() # agregar en el helper de los filtros
    ##rmfDataLog = rmfDataLog[rmfDataLog.monetary > 0] # agregar en el helper de los filtos
    plt.figure(figsize=(15,10))
    # Vemos el resultado y comparamos
    plt.subplot(3, 1, 1, title="transformación Logaritma")
    sns.distplot(rmfDataLog['recency']).set_xlabel("recency")
    plt.subplot(3, 1, 2)
    sns.distplot(rmfDataLog['monetary']).set_xlabel("monetary")
    plt.subplot(3, 1, 3)
    sns.distplot(rmfDataLog['frequency']).set_xlabel("frequency")
    return rmf, rmfDataLog

def calcuteSilhouetteMethod(rmfDataLog):
    from sklearn.metrics import silhouette_score
    silhouette = []
    for n_clusters in range(2,10):
        # para cada cluster definido arriba, creamos el modelo de kmeans
        clusterer = KMeans(n_clusters=n_clusters)
        preds = clusterer.fit_predict(rmfDataLog)
        # Calculamos el coeficiemte del método de la silueta. 
        score = silhouette_score(rmfDataLog, preds)
        silhouette.append(score)
        #imprimimos los distintos coeficientes para cada numero de cluster.
        print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
    
    # Gráficamos
    n_clusters = [rmfDataLog for rmfDataLog in range(2,10)]
    df_silhouette = pd.DataFrame({'n_clusters':n_clusters,'silhouette_score':silhouette})
    sns.lineplot(data=df_silhouette, x="n_clusters", y="silhouette_score")
    
    
    
        
    fig, ax = plt.subplots(4, 2, figsize=(15,8))
    plt.setp(ax, xlim=(0,0.6))
    for i in [2, 3, 4, 5, 6, 7, 8, 9]:
        '''
        Create KMeans instance for different number of clusters
        '''
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        
        # obtenemos el cociente(q) y el resto(m) para cada n
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        ax=ax[q-1][mod] -> see image in the document.
        '''
        
        # Viendo los resultados 
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(rmfDataLog)

def calculateKmeans(rmf):
    kmeans = KMeans(n_clusters=4, random_state=1)

    # entrenamos con el metodo k-means agrupando con la informacion de la tabla anteriormente creada.
    kmeans.fit(np.log(rmf[['recency','frequency', 'monetary']]))

    # Obtenemos la columna de la clasificación mediante el atributo '.labels_'
    cluster = kmeans.labels_
    
    rmf = rmf.assign(cluster = cluster)
    
    clustersData = rmf.groupby(['cluster']).agg({'recency': 'mean',
                                                    'frequency': 'mean',
                                                    'monetary': ['mean', 'count'],}).round(0)

    # agregamos esta nueva columna a tabla anterior
    return rmf, clustersData

def addSegmentations(rmf):
    
    rfm = rmf[['recency','frequency','monetary','cluster', 'std', 'media', 'quantity']]
    
    rfm["recencyScore"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequencyScore"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    # 1 1,1,2,3,3,3,3,3,
    rfm["monetaryScore"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["rfScore"] = (rfm['recencyScore'].astype(str) +
                        rfm['frequencyScore'].astype(str))
    
    
    rfm["frecuencyMonetaryScore"] = rfm['frequencyScore'].astype(int) + rfm['monetaryScore'].astype(int)
    rfm["frecuencyMonetaryScore"] = pd.qcut(rfm['frecuencyMonetaryScore'], 5, labels=[1, 2, 3, 4, 5])
    rfm["rfScore"] = (rfm['recencyScore'].astype(str) + rfm['frecuencyMonetaryScore'].astype(str))
    rfm.head()
    # creamos el mapa que nos va a decir con respecto a la ubicacion del cliente 
    # dado por las coordenas x,y su correspondiente label.
    seg_map = {
        r'[1-2][1-2]': 'perdidos',
        r'[1-2][3-4]': 'Hibernando',
        r'[1-2]5': 'No puedes perderlos',
        r'3[1-2]': 'Durmiendose',
        r'33': 'Necesitan atención',
        r'[3-4][4-5]': 'Fieles',
        r'41': 'Promesas',
        r'51': 'Clientes nuevos',
        r'[4-5][2-3]': 'Potencial fieles',
        r'5[4-5]': 'Campeones'
    }
    # aplicamos el mapeo
    rfm['segment'] = rfm['rfScore'].replace(seg_map, regex=True)
    return rfm
def getGroupByInvoiceClients(dataSet):
    df = dataSet
    df['totalValue']= df['quantity'] * df['price']
    df = df[df.totalValue > 0]

    # Agrupamos por la factura y seleccionamos los campos necesarios.
    df= df.groupby('invoice').agg({'totalValue': 'sum', 'invoiceDate': 'first', 'itemId': 'count', 'country':'first', 'customerId':'first', 'quantity':'sum' })

    ## Utilizamos la cantidad de items como la cantidad de productos en un pedido.
    df = df.rename(columns={'itemId': 'products'})
    df['invoice']= df.index.astype('int')
    clients = df.groupby("customerId").agg({"totalValue": "sum", "invoice": "count"})
    #clientes = df[df.totalValue>20000]

    clients = clients[clients.invoice > 50]
    return df, clients.sort_values(by="invoice", ascending= False)
    
def singleRegressionLineal(df, clients):
   
    for client in clients.index[0:3]:
        resultRegression  = regressionModel.runRegression(df, client)
        
        # Visualizamos en dos graficos los resutlados de la regresión.
        info.viewResultRegression(resultRegression)
        x_train, x_test, y_train, y_test, y_train_pred , y_test_pred = resultRegression 
        
        data = Data(x_train, x_test, y_train, y_test, y_test_pred)
        info.modelLoss(data)
    


def singleRegressionLinealByMonth(df, clients):
    for client in clients[0:3].index: ## TODO aparentemente poco preciso
        resultRegression = regressionModel.runRegressionByMonth(df, client)
        info.viewResultRegressionByMonth(resultRegression, client)
        
        x_train, x_test, y_train, y_test , y_test_pred , y_train_pred= resultRegression 
        data = Data(x_train, x_test, y_train, y_test, y_test_pred)
        info.modelLoss(data)
        
        
        
def runLinealPredictions(groupByInvoice):
   
    labels = Labels("Facturado", "cantidad", "Regresión Lineal\n Linea Azul: Predicción\n Punto rojo: Entrenamiento\n Punto Azul: Test")
    result = regressionModel.runRegressionByOrder(groupByInvoice,  'price', "quantity")
    info.viewResultRegressionByOrder(result, labels, False)
    info.modelLoss(result)
            
    labels = Labels("Facturado", "Variedad", "Regresión Lineal\n Linea Azul: Predicción\n Punto rojo: Entrenamiento\n Punto Azul: Test")
    result = regressionModel.runRegressionByOrder(groupByInvoice,  'price', "variety")
    info.viewResultRegressionByOrder(result, labels, False)
    info.modelLoss(result)
    
    labels = Labels("Cantidad", "Variedad", "Regresión Lineal\n Linea Azul: Predicción\n Punto rojo: Entrenamiento\n Punto Azul: Test")
    result = regressionModel.runRegressionByOrder(groupByInvoice,  'quantity', "variety")
    info.viewResultRegressionByOrder(result, labels, False)
    info.modelLoss(result)
    
    
def prepareDataToRegression(df, segmentation):
    df.invoiceDate = df.invoiceDate.apply(lambda x: datetime(x.year, x.month, x.day))
    # Utilizamos la fecha de la factura como índice
    df.index = df.invoiceDate
    df = df.drop(columns = ['invoiceDate'])
    df= df.sort_index()
    #df['orders'] = 1
    #df= df.groupby('invoiceDate').agg({"totalValue": 'sum', 'products': 'sum', "orders": 'sum'})
    df['date']= df.index
    df['week']= calculateWeek(df.date)
    
    countries = df[['country']]
    ## TODO Operación muy lenta, verificar como mejorar
    listCountries = loadCountries()
    df['country'] = countries.country.apply(lambda x: getCountryCode(x, listCountries) )
    
    # testeo modelo sacando los casos extremos
    df = df[df.totalValue < 12000]
    
    df =df.drop(columns=['date'])
    
    #XX = df.drop(columns=['totalValue'])
    
    # aplicamos el mapeo
    segmentation['segmentCode'] = segmentation['rfScore'].replace(constants.seg_map, regex=True)
    df['id']= df.customerId
    segmentation['client'] = segmentation.index
    #segmentation = segmentation[['segmentCode','client', 'cluster']]
    df['segmentation'] =df.id.apply(lambda x: getSegmentation(x, segmentation))
    df['cluster'] = df.id.apply(lambda x: getCluster(x, segmentation))
    df['avgPrice'] = df.id.apply(lambda x: getAvgPrice(x, segmentation))
    df['std'] =  df.id.apply(lambda x: getStd(x, segmentation))
    df['media'] =  df.id.apply(lambda x: getMedia(x, segmentation))
    df['totalItems']= df.id.apply(lambda x: getAvgItems(x,segmentation))
    
    df['orderQuantity'] = df.id.apply(lambda x: getFrequency(x,segmentation))
    df['avgQuantity'] = df['totalItems'] / df['orderQuantity']
    
    df = df[df['totalValue'] < df['media'] + 3 * df['std']]
    
    features = ['week', 'products', 'customerId', 'country', 'segmentation', 'invoice', 'cluster', 'avgPrice', 'avgQuantity']
    value = ['totalValue']
    return df,features,value 

def getBestProducts(dataSet):
    bestProduct = dataSet.groupby(dataSet.itemId.tolist(), as_index=False, dropna=False).size().rename(columns={'index':'itemId', 'size': 'quantity'})
    ## ordenamos y guardamos los primeros 20
    bestProduct = bestProduct.sort_values('quantity', ascending=False)[0:20]
    print( bestProduct.head(10))
    bestProduct.to_csv('bestProduct.csv')


def getCountryCode(x, listCountries):
    match = listCountries.loc[listCountries['name'] == x]
    if match.size > 0:
        return match.countryCode.values[0]
    else:
        return  0
    
def getSegmentation(x, segmentation):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.segmentCode.values[0]
    else:
        return 404


def getCluster(x, segmentation):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.cluster.values[0]
    else:
        return 404
    
def getFrequency(x, segmentation):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.frequency.values[0]
    else:
        return 404
    
def getAvgPrice(x,segmentation):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.monetary.values[0]/match.frequency.values[0]
    else:
        return 0
    
def getStd(x,segmentation):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match['std'].values[0]
    else:
        return 0

def getMedia(x,segmentation):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.media.values[0]
    else:
        return 0
def getAvgItems(x, segmentation):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.quantity.values[0]
    else:
        return 404
    
class Labels(object):
    def __init__(self, x, y, title):
        self.x = x
        self.y = y
        self.title = title
        
    