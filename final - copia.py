# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:04:49 2022

@author: Luciano
"""

import tensorflow as tf
import helpers
import info
import regressionModel
import compareModels
from datetime import datetime
import pandas as pd


print(tf.__version__)

class Data(object):
    def __init__(self, x_train, x_test, y_train, y_test, y_test_pred= None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_pred =  y_test_pred
        


# Con la base de datos ya preparada, la vamos a leer y luego asignar los nombres de las columnas.
dataSet = helpers.loadDataSet()

# Leer informacion básica de la base de datos.
info.initialInfo(dataSet)

# Aplicamos filtros necesarios a la base de datos.
dataSet = helpers.filterDataSet(dataSet)



# Análisis de cohort

helpers.cohortAnalysis(dataSet)
## TODO Sacar un analisis del cohortToPlot !!!!!!!!!!!!!!!!!!!

# Calculamos los valores de RMF
rmf, rmfDataLog = helpers.calculateRMF(dataSet)

# Graficamos el metodo del Codo para obtener la cantidad adecuada de numero de clusters (k)
#https://machinelearningparatodos.com/segmentacion-utilizando-k-means-en-python/#:~:text=El%20m%C3%A9todo%20del%20codo%20utiliza,las%20observaciones%20a%20su%20centroide.&text=Cuanto%20menor%20es%20la%20distancia,la%20distancia%20media%20intra%2Dcluster.

info.showElbowMethod(rmfDataLog)

# interpretamos que podriamos tomar como el punto de inflexion un valor de k entre 2 y 4.

# Con este método no nos fue suficiente, procederemos a hacer el metodo de la silueta


helpers.calcuteSilhouetteMethod(rmfDataLog)

           
# TODO análisis - Luego del analisis, vamos a usar el nro de cluster igual a 4.
# entrenamos con el metodo k-means agrupando 
rmf, clustersData = helpers.calculateKmeans(rmf)
## TODO Guardar esta variable clustersData en el documento

# Mostramos en un gráfico la importancia de cada Cluster con respecto al rmf
info.showClusterImportance(rmf)

'''
Otro analisis de los clientes que podemos realizar es el analisis 
de la sumatoria de la frecuencia y el valor monetario con respecto a la 
recencia.
'''

# Evaluaremos para cada cliente su segmentación.
rmf = helpers.addSegmentations(rmf)

# Graficamos todas las segmentación de los clientes.
info.showSegmentation(rmf)



# predecir proximo pedido


"""
Vamos a aplicar un modelo de regresion lineal.
sobre la cantidad de items vs precio de la factura.
"""
df, clients = helpers.getGroupByInvoiceClients(dataSet)

#helpers.singleRegressionLineal(df, clients)


"""
Con la información obtenida hasta el momento, intentaremos crear una 
regresión lineal para los principales clientes
considerando el total de lo facturado a un cliente en un mes calendario.
Se convierte los datos de los meses en formato de calendario gregoriano que es numerico
así se puede entrenar el modelo.
"""

#helpers.singleRegressionLinealByMonth(df, clients)

    

"""
Aplicaremos regresion lineal a cada pedido con el fin de predecir algunas cosas
Variedad de productos en el pedido
Precio total de la factura
Cantidad de productos totales.
"""




groupByInvoice = helpers.groupByInvoice(dataSet)

"""
Realizamos algunas predicciones lineales.

"""
#helpers.runLinealPredictions(groupByInvoice)


segmentation = rmf 




"""
Preparamos los datos para correr entrenar el modelo de tensorFlow
importante https://www.cienciadedatos.net/documentos/py35-redes-neuronales-python.html
"""

df2, features, value = helpers.prepareDataToRegression(df, segmentation)

## Vamos a proceder a hacer pruebas con distintos parámetros para el modelo para encontrar el óptimo.
epochs = 1000

"""
Seleccíon de la funcion de activación
"""
#compareModels.selectActivationFunction(df2,features, value)
# Elección: RELU

"""
Seleccíon del metódo de inicialización
"""
#compareModels.selectInitializationMethod(df2,features, value)
# Elección: RandomUniform

"""
Seleccíon del metódo de activación
"""
#compareModels.selectActivationMethod(df2,features, value)

# Elección: ADAM


""" Teniendo ya los distintos parametros que modemos confiugurar para un modelo, procedemos a 

Utilizar funcion de activación RELU
Utilizar metodo de inicializacion de adam
Utilizar  RandomUniform como inicializador de kernel
se vio que un entrenamiento de 3000 epocas suele ser suficiente, ya que luego el aprendizaje es muy bajo, igualmente como este 
entrenamiento es por unica vez, vamos a realizarlo con 5000 epocas, para tener un margen.

# Por último vamos a comparar modelos con distinas cantidad de capas y de tamañp
# Lo haremos con una cantidad de epocas alta (5000)

"""

#compareModels.selectSizeForModel(df2, features, value,epochs)
# Elección: modelo mediano con 5000 épocas





"""predecir para un cliente de los mas grandes. *********************************************"""
#compareModels.predictOneClient(df2, features, value)
# Modelo con una performance bastante inferior al anterior.


""" Usar la predicción de todos los clienes para predecir y comparar para uno de los mejores clientes ***************"""
#compareModels.predictOneClientWithBestModel(df2[df2.customerId == 14911], mediumModelCreated,  historyMedium, features, value)



mediumData, mediumModelCreated, historyMedium = regressionModel.runPrediction(df2, features, value,5000, "medium")

## Filtramos y Guardamos los datos en un archivo.
dataToExport  = helpers.exportResultPrediction(mediumData)

## Agregamos nombres falsos a los clientes, ya que la base de datos no los contiene
## Máximo datos a exportar son 999 (limitados por los nombres)
dataToExport = dataToExport[0:999]
dataToExport = helpers.loadFakesNames(dataToExport)
dataToExport.head(1)

## procedemos a calcular los productos mas repetidos en distintas ordenes para cada cliente, ordenamos y solicitamos los 5 
## fundamentales

dataToExport = helpers.addBestProducts(dataSet, dataToExport)



## Guardamos en la base de datos la base trabajada de los clientes como de los productos
dataToExport.to_csv('dataToExport.csv')



## ## Calculamos los productos mas vendidos y guardamos el archivo bestProduct.csv
helpers.getBestProducts(dataSet),


## Repetimos los productos mas vendidos pero del ultimo més guardando el archivo bestProductLastMonth.csv
helpers.getBestProductsLastMonth(dataSet)


## Exportamos una tabla con las caracteristicas de cada producto.
helpers.exportItemsDetails(dataSet)

"Ver los pedidos mensuales de un producto."



## Comparar e mostrar con otros modelos los cuales fueron desechados.

compareModels.compareAllModels()
