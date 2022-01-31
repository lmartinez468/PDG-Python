# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:29:19 2021

@author: Luciano
"""
import pandas as pd
from IPython.display import display
from collections import Counter


def loadDataSet():
    url = "file:///C:/Users/Luciano/Documents/pdg/agosto21comaFinal.csv"
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
        
    dataSet.reset_index(drop=True,inplace=True)
    
    # rmfData = rmfData.dropna() # agregar en el helper de los filtros
    # rmfData = rmfData[rmfData.monetary > 1] # agregar en el helper de los filtos
    
    # Como encontramos devoluciones procederemos a eliminarlas del dataSet (TODO ¿ elimnar la factura asociada ?)
    return dataSet[~negativeQuantity]
    display("resultado",dataSet)
    
    
def groupByInvoice(dataSet):
     dataSet['variety'] = 1
     dataSet['price'] = dataSet['price'] * dataSet['quantity']
     return dataSet.groupby(dataSet['invoice'], as_index = False).agg({
         "invoiceDate": "first",
         "price": 'sum',
         'customerId': 'first',
         'variety': 'count',
         'quantity': 'sum'
         })

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
    dataToExport["totalValuePredicted"] = data.y_pred
  
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

def loadFakesNames(data):
    fakesNames= loadNames()
    data['name']= fakesNames.name.values[0:len(data)]
    return data

def loadNames():
    url = "file:///C:/Users/Luciano/.spyder-py3/autosave/PDG/names.csv"
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
    allItems.to_csv('allItems.csv')