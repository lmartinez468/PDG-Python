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

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

"""# **Importing Datasets**"""
from sklearn.model_selection import train_test_split 
from collections import Counter

def calculateWeek(date):
    years= date.dt.strftime('%Y').astype(int) - 2009
    weeks= date.dt.strftime('%W').astype(int)
    return years * 51 + weeks

print(tf.__version__)

class Data(object):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test 
        


# Con la base de datos ya preparada, la vamos a leer y luego asignar los nombres de las columnas.
dataSet = helpers.loadDataSet()

# Leer informacion básica de la base de datos.
info.initialInfo(dataSet)

# Aplicamos filtros necesarios a la base de datos.
dataSet = helpers.filterDataSet(dataSet)



# Como vemos las órdenes(invoice) en el dataSet se encuentran desglosada por cada item que se encuentra en el pedido
# Agrupamos todos los productos del pedido, obtentiendo solamente el total de la orden y la cantidad de productos.
#dataSetByInvoice = helpers.groupByInvoice(dataSet)

# Agrupamos el total de ordenes para cada cliente.
#groupByCustomer = helpers.groupByCustomer(dataSetByInvoice)

#groupByCustomer['customerId'] = groupByCustomer.customerId.astype(str) ## TODO mover esta conversion al principio.

# Vemos la cantidad de pedidos de cada cliente y así poder determinar cuales clientes son adecuados para el análisis,
# ya que es necesario un nro elevado de pedidos.
#info.ordersByCLient(groupByCustomer)


## TODO complementar con análisis de media, mediana.
## Analizaremos los mejores 200 clientes ( considerandolo por la cantidad de pedidos).
#better200 = groupByCustomer[0:200]

# Visualizamos los mejores clientes.
#info.viewBetterClients(better200)

# Visualizamos para los mejores clientes, los pedidos por mes.
#info.viewOrdersByMonths(better200, dataSetByInvoice)


"""

# inserte aquiii 


"""

"""
Realizaremos un analisis de cohort 
"""

# primeros sobre la base de datos tenemos que averiguar para cada cliente, la fecha del primer pedido que hizo, que lo agregaremos
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

## procedemos a agregar otro nueva columna llamada 'diff', que seria la diferencia de dias entre el primer pedido y el pedido.
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
cohortToPlot =cohortToPlot.round(3) * 100


# Graficamos
plt.figure(figsize=(30, 8))
plt.title('Ratio de usuarios que vuelven a comprar')
sns.heatmap(data = cohortToPlot,annot = True,fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap = 'BuGn')
plt.show()

## TODO Sacar un analisis del cohortToPlot !!!!!!!!!!!!!!!!!!!


## verificamos la fecha del ultimo pedido que tomariamos como fin del dataSet
# se agrega +1, para utilizar valores igual o mayores a 0, ya que 0 luego no se puede utilizar el logartimo
endDate = dataSet['invoiceDate'].max() + timedelta(1)

## Creamos una nueva columna con la multiplicacion del precio del producto por la cantidad.
dataSet['totalValue']= dataSet['quantity'] * dataSet['price']

## Creamos una nueva variable agrupando el dataSet por cliente y creando las 3 columnas que nos interesan para este analisis que son las 
## frecuencia, recencia y valor monetario.


## Removemos los valores negativos o igual a 0.


## fixear con esto
data2 = dataSet.groupby(['invoice']).agg({'invoiceDate': 'first',
                                                            'invoice': 'first',
                                                             'totalValue': 'sum',
                                                             'customerId': 'first'})

data2 = data2.groupby(['customerId']).agg({'invoiceDate': lambda x: (endDate - x.max()).days,
                                                            'invoice': 'count',
                                                             'totalValue': 'sum',
                                                             }).rename(columns={'invoiceDate': 'recency',
                                                        'invoice': 'frequency','totalValue': 'monetary'})                                                                                                     
                                                                                          
data2 = data2[data2.monetary > 1]

data2['monetary'] = data2['monetary'].apply(lambda x: int(round(x)))    



## para cada uno de estas 3 nuevas columnas, vamos a segmentarlas

## Para poder saber el grado de segmentacion xxxxxxxxxxxxxxxxxxxxxxxxxx
## introducir acaaa 

# Crearemos una variable donde calcularemos los cuartiles de cada columna 
# ascendingLabels donde un cliente con valor numerico mas bajo es favorable, ya que la última transaccion fue reciente, por lo que invertiremos el start y stop.
# ejemplo valores de recencia de [10,30,20,40] -> corresponde a cuantiles [4,3,2,1]
invertLabels = range(4, 0, -1)
recencyQuartiles = pd.qcut(data2['recency'], 4, labels = invertLabels)

# para la frecuencia y el valor monetario, no se invertira el start y stop, ya que mientras mas alto sea el valor númerico es mas favorable
normalLabels = range(1,5)


frequencyQuartiles = pd.qcut(data2['frequency'], 4, labels = normalLabels)
monetaryQuartiles = pd.qcut(data2['monetary'], 4, labels = normalLabels)

## Agregamos a la tabla las tres nuevas columnas calculadas
data2 = data2.assign(R = recencyQuartiles.values,
                     F = frequencyQuartiles.values,
                     M = monetaryQuartiles.values,
                     )
## Agregamos una nueva columna concatenando RMF, donde el intervalo sera de 111 a 444, donde minetras mayor sea el valor, el cliente es mas importante
data2 = data2.assign(RMF = data2['R'].astype(str) + data2['F'].astype(str) + data2['M'].astype(str))
def pepe(a,b,c):
    return a.values +b.values +c.values
    
data2.head()
#sns.lineplot(x="customerId", y="R", hue='Cluster', data=data2)
plt.figure()
plt.subplot(3, 1, 1)
sns.distplot(data2['recency'])
plt.subplot(3, 1, 2)
sns.distplot(data2['monetary'])
plt.subplot(3, 1, 3)
sns.distplot(data2['frequency'])


# Aplicamos una transformación logaritmica.
rmfData = data2[['recency','frequency','monetary']]
#rmfData = rmfData.dropna() # agregar en el helper de los filtros
#rmfData = rmfData[rmfData.monetary > 1] # agregar en el helper de los filtos
rmfDataLog =  np.log(rmfData)
##rmfDataLog = rmfDataLog.dropna() # agregar en el helper de los filtros
##rmfDataLog = rmfDataLog[rmfDataLog.monetary > 0] # agregar en el helper de los filtos
plt.figure()
# Vemos el resultado y comparamos
plt.subplot(3, 1, 1, title="transformación Logaritma")
sns.distplot(rmfDataLog['recency']).set_ylabel("recency")
plt.subplot(3, 1, 2)
sns.distplot(rmfDataLog['monetary']).set_ylabel("monetary")
plt.subplot(3, 1, 3)
sns.distplot(rmfDataLog['frequency']).set_ylabel("frequency")


# Con el metdo del codo tratatermos de interpetrar los numeros de cluster
from sklearn.cluster import KMeans

distortions = []
K = range(2,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(rmfDataLog)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# interpretamos que podriamos tomar como el punto de inflexion un valor de k entre 2 y 4.


# Con este método no nos fue suficiente, procederemos a hacer el metodo de la silueta

from sklearn.metrics import silhouette_score, silhouette_samples
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
    
    
    
   
# Luego del analisis, vamos a usar el nro de cluster igual a 4.

kmeans = KMeans(n_clusters=4, random_state=1)

# entrenamos con el metodo k-means agrupando con la informacion de la tabla anteriormente creada(rmfDataLog).
kmeans.fit(rmfDataLog)

# Obtenemos la columna de la clasificación mediante el atributo '.labels_'
cluster = kmeans.labels_


# agregamos esta nueva columna a tabla anterior

rmfData = rmfData.assign(cluster = cluster)
#rmfDataLog = rmfDataLog.assign(cluster = cluster)

# Create a cluster label column in the original DataFrame
# data_norm_k4 = rmfDataLog.assign(Cluster = cluster_labels)
# data_k4 = rmfData.assign(Cluster = cluster_labels)



clustersData = rmfData.groupby(['cluster']).agg({'recency': 'mean',
                                                    'frequency': 'mean',
                                                    'monetary': ['mean', 'count'],}).round(0)





## TODO Guardar esta variable en el documento
display(clustersData)


# Agregamos a la tabla una columna con el id de los clientes.


##rmfData.index = data2['customerId']

rmfData.head()

# Hemos logrado crear una tabla con RMF y el grupo pertenenciante este cliente, por lo cual tenemos clasificadoa  cada cliente



# agrupamos todos los clientes por grupo y obtenemos la media de todos los clusters
# y asi podemos sacar conclusion de cada cliente.



clusterGrouped = rmfData.groupby(['cluster']).mean()


display(clusterGrouped)
plt.plot(clusterGrouped.monetary, clusterGrouped.index )
plt.ylim([0,2000])
rmfDataMean = rmfData.mean()

# calculamos la impirtancia relativa en cada grupo

relative_imp = clusterGrouped / rmfDataMean - 1
relative_imp.round(2)





relative_imp = relative_imp.drop(['cluster'], axis=1)
# Plot heatmap
plt.cla()
plt.close()
rmfData
plt.figure(figsize=(20, 20))
plt.title('Importancia de los atributos')
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
sns.set(font_scale=3)
## TODO hacer análisis de este gráfico en el documento
plt.show()


##from



'''
Otro analisis de los clientes que podemos realizar es el analisis 
de la sumatoria de la frecuencia y el valor monetario con respecto a la 
recencia.
'''

# creamos una nueva tabla llamada rmf, donde asiganmos valores del 1 a 5 a 
# cada columna.
# ademas creamos una columna con la sumatoria de la recencia y la frecuencia.


rfm = rmfData
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
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
# aplicamos el mapeo
rfm['segment'] = rfm['rfScore'].replace(seg_map, regex=True)
# Guaramos en una nueva variable el resultados de lo calculado anteriormente
analysisResult = rfm[['segment', 'cluster']]

# TODO - rename columns name
clientsBySegments = rfm.groupby(rfm.segment).agg({'monetary': 'count'} )


# Empezamos a crear el gráfico de torta
fig, ax = plt.subplots(figsize=(26, 26))

clients = clientsBySegments.monetary
segments = clients.index



def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


wedges, texts, autotexts = ax.pie(clients, autopct=lambda pct: func(pct, clients),
                                  textprops=dict(color="black"), radius= 1.2, 
                                  colors=['#ff0000','#fe4400','#fa6500', 
                                          '#64ff00','#f28100','#e69a00',
                                          '#d7b100','#c4c600','#adda00',
                                          '#8fed00','#64ff00'])


ax.legend(wedges, segments,
          title="Segments",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1), fontsize=30)
plt.setp(autotexts, size=25, weight="bold")
plt.rcParams['font.size'] = 100
ax.set_title("Distribucion de clientes en cada segmentación")
## TODO usar e interpretar gráfico
plt.show()



# predecir proximo pedido

'''
primeros necesitamos obtener t, que es la antiguedad de cada cliente.
]

'''
t =dataSet.groupby(['customerId']).agg({'invoiceDate': lambda x: (endDate - x.min()).days, 'totalValue': 'sum'})
t = t[t.totalValue > 1]
cltv_df = rfm[['frequency', 'monetary', 'recency']]
cltv_df = cltv_df.assign(t=t.invoiceDate)



bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['t'])

# Applying the model to the data set and adding as a variable
# Calculation of expected 1 month of purchase
cltv_df["expected_purc_1_month"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['t'])
cltv_df.head()






ggf = GammaGammaFitter(penalizer_coef=0.01)
cltv_df = rfm[['frequency', 'monetary', 'recency']]
# agregamos t a la tabla existente
cltv_df.assign(t=t.invoiceDate)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# Applying the model to the dataset
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

# Adding model as a variable 
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.head()



"""
Para un determinado cliente, vamos a aplicar un modelo de regresion lineal.
sobre la cantidad de items vs precio de la factura.
"""
df = dataSet
df['totalValue']= df['quantity'] * df['price']
df = df[df.totalValue > 0]

# Agrupamos por la factura y seleccionamos los campos necesarios.
df= df.groupby('invoice').agg({'totalValue': 'sum', 'invoiceDate': 'first', 'itemId': 'count', 'country':'first', 'customerId':'first' })

## Utilizamos la cantidad de items como la cantidad de productos en un pedido.
df = df.rename(columns={'itemId': 'products'})
df['invoice']= df.index.astype('int')
clientes = df.groupby("customerId").agg({"totalValue": "sum", "invoice": "count"})
#clientes = df[df.totalValue>20000]

clientes = clientes[clientes.invoice > 50]
for client in clientes.index[0:10]:
    resultRegression  = regressionModel.runRegression(df, client)
    
    # Visualizamos en dos graficos los resutlados de la regresión.
    info.viewResultRegression(resultRegression)


"""
Con la información obtenida hasta el momento, intentaremos crear una 
regresión lineal para los principales clientes
considerando el total de lo facturado a un cliente en un mes calendario.
Se convierte los datos de los meses en formato de calendario gregoriano que es numerico
así se puede entrenar el modelo.
"""
for client in clientes[0:1].index: ## TODO aparentemente poco preciso
    resultRegression = regressionModel.runRegressionByMonth(df, client)
    info.viewResultRegressionByMonth(resultRegression, client)

"""
Aplicaremos regresion lineal a cada pedido con el fin de predecir algunas cosas
Variedad de productos en el pedido
Precio total de la factura
Cantidad de productos totales.
"""
# Primero tomaremos al cliente mas grande 14911
dataSetByClient = dataSet[dataSet.customerId == 17920]
ordersByClient = helpers.groupByInvoice(dataSetByClient)

class Labels(object):
    def __init__(self, x, y, title):
        self.x = x
        self.y = y
        self.title = title
        

# convierto a formato gregoriano la fecha para poder utilizarla en la regresión.
ordersByClient['invoiceDate'] = ordersByClient['invoiceDate'].apply(datetime.toordinal)    

# regression para el total de la factura
labels = Labels("fecha", "precio", "Regresión Lineal\n Linea Azul: Predicción\n Punto rojo: Entrenamiento\n Punto Azul: Test")
result = regressionModel.runRegressionByOrder(ordersByClient, 'price', "invoiceDate")
info.viewResultRegressionByOrder(result, labels, True)

# regression para la variedad total de la factura

labels = Labels("fecha", "variedad de productos", "Regresión Lineal\n Linea Azul: Predicción\n Punto rojo: Entrenamiento\n Punto Azul: Test")
result = regressionModel.runRegressionByOrder(ordersByClient, 'variety', "invoiceDate")
info.viewResultRegressionByOrder(result, labels, True)

# regression para el total de la factura

labels = Labels("fecha", "Cantidad de productos", "Regresión Lineal\n Linea Azul: Predicción\n Punto rojo: Entrenamiento\n Punto Azul: Test")
result = regressionModel.runRegressionByOrder(ordersByClient, 'quantity', "invoiceDate")
info.viewResultRegressionByOrder(result, labels, True)

# regression para el total de la factura con respecto a cantidad 
labels = Labels("Precio", "Cantidad de productos", "Regresión Lineal\n Linea Azul: Predicción\n Punto rojo: Entrenamiento\n Punto Azul: Test")
result = regressionModel.runRegressionByOrder(ordersByClient, 'quantity', "price")
info.viewResultRegressionByOrder(result, labels, False)



## NEEEEEEEEEEEEEEEW DATA
## Utilizamos  los Códigos de los paises obtenidos de la ISO
def loadDataSet():
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
    return data[['name', 'countryCode']]

listCountries = loadDataSet()

## Necesitamos renombrar 1 Sólo pais para que coincida con la base de datos
listCountries['name']= listCountries.name.replace(to_replace ="United States",
                 value ="USA")
def getCountryCode(x):
    match = listCountries.loc[listCountries['name'] == x]
    if match.size > 0:
        return match.countryCode.values[0]
    else:
        return  0

segmentation = rmfData

seg_map = {
    r'[1-2][1-2]': 0, #hibernating
    r'[1-2][3-4]': 1, #at_Risk
    r'[1-2]5': 2, #cant_loose
    r'3[1-2]': 3, # about_to_sleep
    r'33': 4, # need_attention
    r'[3-4][4-5]': 5, #loyal_customers
    r'41': 6, #promising
    r'51': 7, #new_customers
    r'[4-5][2-3]': 8, #potential_loyalists
    r'5[4-5]': 9# champion
}


def getSegmentation(x):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.segmentCode.values[0]
    else:
        return 404


def getCluster(x):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.cluster.values[0]
    else:
        return 404
    
def getAvgPrice(x):
    match = segmentation.loc[segmentation['client'] == x]
    if match.size > 0:
        return match.monetary.values[0]/match.frequency.values[0]
    else:
        return 0

def getAvgItems(x):
    match = df.loc[df['customerId'] == x]
    if match.size > 0:
        return match.products.values[0]
    else:
        return 404
    


"""# **Data Splitted into Training, Validation, Test**"""
## Volvemos a utilizar el dataSet


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
df['country'] = countries.country.apply(lambda x: getCountryCode(x) )

# testeo modelo sacando los casos extremos
df = df[df.totalValue < 12000]

df =df.drop(columns=['date'])

#XX = df.drop(columns=['totalValue'])

# aplicamos el mapeo
segmentation['segmentCode'] = segmentation['rfScore'].replace(seg_map, regex=True)
df['id']= df.customerId
segmentation['client'] = segmentation.index
#segmentation = segmentation[['segmentCode','client', 'cluster']]
df['segmentation'] =df.id.apply(lambda x: getSegmentation(x))
df['cluster'] = df.id.apply(lambda x: getCluster(x))
df['avgPrice'] = df.id.apply(lambda x: getAvgPrice(x))
df['totalItems']= df.id.apply(lambda x: getAvgItems(x))

df['orderQuantity'] = 1
df['avgQuantity'] = df['totalItems'] / df['orderQuantity']

features = ['week', 'products', 'customerId', 'country', 'segmentation', 'invoice', 'cluster', 'avgPrice', 'avgQuantity']
value = ['totalValue']

## predecir todos clientes **********************************************
## R2 0.41860542944866796
modelCreated, history, data = regressionModel.runPrediction(df, features, value)

data.y_pred = modelCreated.predict(data.x_test)

info.modelLoss(history, data)
info.show100Prediction(data)
info.comparePredictionWithOrders(data)
info.compare100PredictionWithOrders(data)


## predecir para un cliente *********************************************
modelCreated2, history2, data2 = regressionModel.runPrediction(df[df.customerId == 14911], features, value)

data2.y_pred = modelCreated2.predict(data2.x_test)

info.modelLoss(history2, data2)
info.show100Prediction(data2)
info.showAllPrediction(data2)
info.comparePredictionWithOrders(data2)



## Usar la predicción de todos los clienes para predecir y comparar para la de un cliente ***************
data3 = data2
data3.y_pred = modelCreated.predict(data2.x_test)
info.show100Prediction(data3)
info.showAllPrediction(data3)
info.comparePredictionWithOrders(data3)


## Filtramos y Guardamos los datos en un archivo.
dataToExport  = helpers.exportResultPrediction(data)

## Agregamos nombres falsos a los clientes, ya que la base de datos no los contiene
## Máximo datos a exportar son 999 (limitados por los nombres)
dataToExport = dataToExport[0:999]
dataToExport = helpers.loadFakesNames(dataToExport)

## procedemos a calcular los productos mas repetidos en distintas ordenes para cada cliente, ordenamos y solicitamos los 5 
## fundamentales

## Guardamos en la base de datos la base trabajada de los clientes como de los productos
dataToExport.to_csv('dataToExport.csv')

## Calculamos los productos mas vendidos
bestAllProducts = helpers.bestAllProducts(dataSet)
## Exportamos los mejores productos en una tabla
bestAllProducts.to_csv('bestProduct.csv')


## procedemos a ver calcular los productos mas repetidos en distintas ordenes globalmente

bestProduct = dataSet.groupby(dataSet.itemId.tolist(), as_index=False, dropna=False).size().rename(columns={'index':'itemId', 'size': 'quantity'})
## ordenamos y guardamos los primeros 20
bestProduct = bestProduct.sort_values('quantity', ascending=False)[0:20]
##bestProduct.to_csv('bestProduct.csv')

## Exportamos una tabla con las caracteristicas de cada producto.
helpers.exportItemsDetails(dataSet)



## copia de predicción pero para solo 1 usuario
## ***********************************************************************************************
14911
12748
17841




## TODO clientes
#1# [OK] - Cargar names a la lista. 
#2# [] - Cargar OBJ con último pedido(date, price, cantidad variedad)
#3# [OK] Segmantacion ( copiar de las acciones)
#4# [OK] - prediccion factura( capaz algun dato mas) 
#5# Acciones a tomar (falta cargar las desc)
#6# productos mas solicitados (Calcular)
#7# OK -estadisticas (
#8#                   promedio costo factura,
#9#                   promedio productos(variedad) comprados, 9.5 cant pedidos
#10#                   fecha ultimo pedido,
#11#                   fecha primer pedido -> df.firstMonthOrder,df.firstYearOrder
#12#                   repetir segmento - df.
 


## TODO productos
#13#Productos mas vendidos
#14#Productos con proyección(prometedores)
#15#Productos en descenso (desapareciendo)
#16#Productos muertos (sin ventas en los ultimos 3 meses)
