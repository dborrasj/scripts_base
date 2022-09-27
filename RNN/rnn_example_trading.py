######################################## LIBRERIAS ##########################################

#Comunes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Preparación de datos
from sklearn.preprocessing import MinMaxScaler


#Modelo de red neuronal
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate




########################################## DATASET ############################################

#Importación de datos de precios de una acción en el tiempo
data = pd.read_csv("data.csv")
print(data.head(5))


#Normalización
sc = MinMaxScaler(feature_range = (0,1))
data_set_scaled = sc.fit_transform(data)


#Creación input para RNN
X =[]
ventana_temp = 30 #pasos de tiempo incluidos en cada muestra de entrenamiento
for j in range(8):
    X.append([])
    for i in range(ventana_temp, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i-ventana_temp:i,j])

X = np.moveaxis(X, [0], [2]) #ventana deslizante

X,yi = np.array(X), np.array(data_set_scaled[ventana_temp:,-1])
y = np.reshape(yi,(len(yi),1))


#División de datos

limite = int(len(X)*0.8)
X_train, X_test = X[:limite], X[limite:]
y_train, y_test = y[:limite], y[limite:]
print(X_train.shape)

##################################################################################

#Construcción del modelo

np.random.seed(10)

lstm_input = Input(shape=(ventana_temp, 8), name = "lstm_input")
inputs = LSTM(150, name = "first_layer")(lstm_input)
inputs = Dense (1, name="dense_layer")(inputs)
output = Activation("linear", name ="output")(inputs)
model = Model(inputs = lstm_input, outputs = output)
adam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer = adam, loss = "mse")
model.fit(x = X_train, y = y_train, batch_size = 15, epochs = 30, shuffle = True, validation_split= 0.1)



#Evaluacion
y_pred = model.predict(X_test)
#for i in range(10):
#    print(y_pred[i], y_test[i])

plt.figure(figsize = (16,8))
plt.plot(y_test, color = "black", label = "Test")
plt.plot(y_pred, color = "green", label = "pred")
plt.legend ()
plt.show()
