import sys
import numpy as np
#import math
import tensorflow as tf



def windowed_dataset(series, window_size, shuffle_buffer,paso=1): 
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, stride=paso, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    # dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def ephem2vect(ephem,
               column,
               num_inputs,
               num_outputs,
               row_initial,
               row_final_train,
               row_final_valid, 
               overlap_fin,
               sampling_period,
               vect_step):
    
    # *****************************************************************
    # COMPROBACIONES
    if column.isalpha() and list(ephem.columns.values) is None:
          sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: ha especificado un nombre de columna pero"+
        "las columnas de la matriz de efemerides no tienen nombres")
          return 0
  
    if column.isalpha() and sum(column == ephem.columns.values) == 0:
          sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: el nombre de columna especificado no"+
        "corresponde a ninguna de las columnas de la matriz de efemerides")
          return 0
  
    if column.isdigit() and (column<str(1) or column>str(ephem.shape[1])):
          sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: el numero de columna especificado"+
        "debe estar comprendido entre 1 y el numero total de columnas"+
        "de la matriz de efemerides")
          return 0
  
    if num_inputs<1:
          sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: num_inputs debe ser mayor que 0")
          return(0)
  
    if num_outputs<1:
          sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: num_outputs debe ser mayor que 0")
          return(0)

    if row_initial<0 or row_initial>ephem.shape[0]:
          sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: el numero de fila inicial especificado "+
        "debe estar comprendido entre 1 y el numero total de filas"+
        "de la matriz de efemerides")
          return(0)
    
    if row_final_valid <=row_initial or row_final_valid>ephem.shape[0]:
              sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: el numero de fila final especificado"+
        "debe ser mayor que el numero de fila inicial, y menor o igual"+
        "que el numero total de filas de la matriz de efemerides")
              return(0)
  
    if abs(sampling_period - round(sampling_period)) > (np.finfo(float).eps**0.5 or
          sampling_period<1):
              sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: sampling_period debe ser un entero mayor que 0")
              return(0)

    if abs(vect_step - round(vect_step)) > (np.finfo(float).eps**0.5 or
          vect_step<1):
              sys.stdout.write(
        "WARNING:\n"+
        "Parametro incorrecto: vect_step debe ser un entero mayor que 0")
              return(0)
  
    
    # *****************************************************************
    # Extraer los datos seleccionados (matriz columna)
    
    # if column!="theta":
    #     sys.stdout.write("WARNING:\n"+
    #     "La variable seleccionada aún no está programada para modelar")
    #     return(0)
    
   # "shuffle_buffer" se va a emplear para mezclar los vectores de manera aleatoria. 
    # Se usa en redes neuronales profundas (No las recurrentes) para evitar un sesgo en
    # el aprendizaje.
    
    shuffle_buffer=7680
    
    # # Se parte la base en entrenamiento, validación y prueba.
    
    train  = ephem[column][:row_final_train]
    train_num = train.shape[0]- num_inputs*sampling_period 
    
    valid  = ephem[column][row_final_train-overlap_fin:row_final_valid]
    valid_num = valid.shape[0]- num_inputs*sampling_period 
        
    ### Se convierten las bases a datos supervisados.
    
    train  = windowed_dataset(train, num_inputs, shuffle_buffer,sampling_period)
    a= np.empty((int(train_num), int(num_inputs))) 
    b= np.empty((int(train_num), 1)) 
    n=0
    for i, j in train:
        a[n,]=i.numpy()
        b[n,]=j.numpy()
        n=n+1
    train=np.concatenate((a,b),axis=1)    
   
    valid  = windowed_dataset(valid, num_inputs, shuffle_buffer,sampling_period)
    a= np.empty((int(valid_num), int(num_inputs))) 
    b= np.empty((int(valid_num), 1)) 
    n=0
    for i, j in valid:
        a[n,]=i.numpy()
        b[n,]=j.numpy()
        n=n+1
    valid=np.concatenate((a,b),axis=1)     
 
    return train,valid
 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
