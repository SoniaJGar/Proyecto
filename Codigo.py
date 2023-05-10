import glob
import numpy as np
import pandas as pd
import os
import wfdb
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from scipy.interpolate import CubicSpline
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft

# FUNCIÓN QUE LIMPIA FRAGMENTOS DE 0'S EN LOS EXTREMOS DE LA SEÑAL

def clean_infal(recordname_new):
    #Leemos las señales de un registro en concreto
    record = pd.read_csv(os.path.join('CSV',recordname_new+'.csv'), header=None)
    record = np.asarray(record)
    print(len(record),record)

    # GuardamoS los índices que no son 0 en la versión aplanada del registro
    # Los índice pares harán referencia a la señal FHR y los impares a la UC
    non_0 = np.flatnonzero(record)

    # Identificamos las posiciones iniciales y finales que son 0's tanto en la
    # señal FHR como en la señal UC y las eliminamos

    # Parte final
    if np.trunc((non_0[-1])/2)<int(len(record)-1):
        record=np.delete(record, range(int(np.trunc((non_0[-1])/2)+1),
                                       int(len(record))),0)
    # Parte inicial
    if non_0[0] >0 and non_0[0] >1:
        record=np.delete(record, range(0,int(np.trunc((non_0[0])/2))),0)
    return record    
        
# FUNCIÓN QUE LIMPIA FRAGMENTOS DE 0'S DENTRO DE LA SEÑAL CUYO TAMAÑO ES SUPERIOR 
# O IGUAL A UN CIERTO UMBRAL

def gap_clean(record, umbral): 
    lista1 = []

    # Recorremos las posciones de la señales en conjunto y guardamos la posición 
    # inicial del inicio de un fragmento con 0's
    for i in range(0,len(record)):
        if np.array_equal(record[i:i+umbral],np.zeros((umbral,2))):
            lista1.append(i)

    # Si se han encontrado coincidencias, guardamos los indices no conecutivos
    if len(lista1)!=0:
        lista2=[lista1[0]]        
        for j in range(0,len(lista1)-1):
            if int(lista1[j])!=int(lista1[j+1])-1:
                lista2.append(lista1[j])
                lista2.append(lista1[j+1])

        lista2.append(lista1[-1])
        print(lista2)

        # Eliminamos los fragmentos desde las posiciones pares a las impares + umbral
        for n in range(1,len(lista2)+1,2):
            record=np.delete(record, range(int(lista2[len(lista2)-n-1]),
                                           int(lista2[len(lista2)-n])+umbral),0)
    return record
    
    
# CONVERTIMOS 0'S A NAN
    # En la señal FHR y UC se convierten a Nan, los valores que son 0, superiores 
    # o inferiores a un umbral

def zeros_to_nan(record, minimo_FHR, minimo_UC, maximo_FHR, maximo_UC):
    record[:,0]=[np.nan if value==0 or value<minimo_FHR or value>maximo_FHR  else value for value in record[:,0]]
    record[:,1]=[np.nan if value==0 or value<minimo_UC or value>maximo_UC else value for value in record[:,1]]
    return record
    
    
#PASO DE ETIQUETAS NUMERICAS A CLASES 0 O 1
    # Devuelve las etiquetas de cada señal y el número de señales con hipoxia y normales
def target(etiquetas, umbral):
    import numpy as np
    y = np.zeros((len(etiquetas),1))
    contador = 0
    for i in range(0,len(etiquetas)):

        # Las señales con una valor inferior a un umbral se etiquetan con 1 
        # Indicador de hipoxia
        if etiquetas[i]<umbral:
            y[i,:] = 1
            contador = contador + 1

        # Las señales con una valor superior a un umbral se etiquetan con 0
        # Indicador de no hipoxia
        else:
            y[i,:] = 0

    pos = contador
    neg = len(y)-contador
    print('Muestras sin hipoxia:', len(y)-contador, (len(y)-contador)/(len(y))*100)
    print('Muestras con hipoxia:', contador, contador/(len(y))*100)
    return y, neg, pos

    
    
# ARQUITECTURA DE RED NEURONAL CNN/LSTM UTILIZADO PARA LOS 3 PRIMEROS MODELOS

def build_cnn_model(unit, kernel, reg):
    model = keras.Sequential()

    # Capa convolucional con Maxpooling
    model.add(keras.layers.Conv1D(unit, kernel, strides=1, activation='relu',
                                  input_shape=input_shape, kernel_regularizer=l2(l=reg))) 
    model.add(keras.layers.MaxPooling1D(2))

    # Capa Flatten
    model.add(keras.layers.Flatten())

    # Capa totalmente conectada
    model.add(keras.layers.Dense(1, activation = 'sigmoid')) 

    # Compilacion del modelo
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), 
                  metrics=['accuracy', 'AUC'])
    return model


def build_LSTM_model(unit, reg):

    model = keras.Sequential()
    # Capa LSTM 
    model.add(keras.layers.LSTM(unit, input_shape=input_shape, activation='relu', 
                                dropout=0.2, recurrent_dropout=0.2))

    # Capa Dropout
    model.add(keras.layers.Dropout(reg))

    # Capa Normalización por lotes
    model.add(keras.layers.BatchNormalization())

    # Capa totalmente conectada
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compilación del modelo
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.00001), 
                  metrics=['accuracy', 'AUC'])
    return model
    
    
# TRANSFORMADA RAPIDA DE FOURIER DE LAS SEÑALES
    
def fourier(matriz):
    N = matriz.shape[1]
    matriz_f_n = np.zeros((len(matriz),N//2, 2))
    for i in range(len(matriz)):
        # Transformamos los datos de la señal FHR y los normalizamos
        close_fft_FHR = np.fft.fft(matriz[i,:,0])/N
        fft_df_FHR = pd.DataFrame({'fft': close_fft_FHR})
        fft_list_FHR = np.asarray(fft_df_FHR['fft'].tolist())

        # Nos quedamos con las frecuencias positivas, es decir, hasta la 
        # frecuecnia de Nyquist
        matriz_f_n[i,:,0] = np.abs(np.copy(fft_list_FHR)[0:N//2])

        # Transformamos los datos de la señal UC y los normalizamos
        close_fft_UC = np.fft.fft(matriz[i,:,1])/N
        fft_df_UC = pd.DataFrame({'fft': close_fft_UC})
        fft_list_UC = np.asarray(fft_df_UC['fft'].tolist())

        # Nos quedamos con las frecuencias positivas, es decir, hasta  
        # la frecuecnia de Nyquist
        matriz_f_n[i,:,1] = np.abs(np.copy(fft_list_UC)[0:N//2])
    return matriz_f_n

# ARUITECTURA NEURONAL CNN/LSTM PROFUNDA
def build_cnn_model_2(unit_1, kernel_1, reg_1, max_1, unit_2, kernel_2, max_2, reg_2):
    model = Sequential()
    # 1ª Capa Convolucional + Maxpooling
    model.add(Conv1D(unit_1, kernel_1, strides=1, activation='relu', 
                     input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(l=reg_1))) 
    model.add(keras.layers.MaxPooling1D(max_1))

    # 2ª Capa convolucional + Maxpooling
    model.add(Conv1D(unit_2, kernel_2, strides=1, activation='relu', 
                     kernel_regularizer=regularizers.l2(l=reg_2)))
    model.add(keras.layers.MaxPooling1D(max_2))

    # Capa Flatten
    model.add(Flatten())

    # Capa Normalización 
    model.add(keras.layers.BatchNormalization())

    # Capa totalmente conectada
    model.add(Dense(1, activation = 'sigmoid'))

    # Compilación del modelo
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), 
                  metrics=['accuracy', 'AUC'])
    return model


def build_LSTM_model_2(unit_1, reg_1, unit_2):
    model = keras.Sequential()
    # 1ª Capa LSTM
    model.add(keras.layers.LSTM(unit_1, input_shape=input_shape, activation='relu', 
                                return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

    # 2ª Capa LSTM
    model.add(keras.layers.LSTM(unit_2, activation='relu'))

    # Capa Dropout
    model.add(keras.layers.Dropout(reg_1))

    # Capa Normalización
    model.add(keras.layers.BatchNormalization())

    # Capa totalmente conectada
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compilación del modelo
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.00001), 
                  metrics=['accuracy', 'AUC'])
    return model


# GRÁFICA DE LA CURVA ROC DE UN MODELO

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=1, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    
