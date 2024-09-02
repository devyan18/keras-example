import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD

# Lectura del archivo CSV
datos = pd.read_csv('altura_peso.csv')

# Visualización del set de datos
datos.plot.scatter(x='Altura', y='Peso')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.show()

# Preparación de los datos para la regresión lineal
x = datos['Altura'].values
y = datos['Peso'].values

# Normalizar los datos (opcional, pero recomendado)
x = x / 100.0  # Normalizamos la altura a metros para una mejor escala
y = y / 100.0  # Normalizamos el peso para mantener una escala similar

# Construir el modelo en Keras
np.random.seed(2)  # Para reproducibilidad del entrenamiento

modelo = Sequential()
modelo.add(Input(shape=(1,)))  # Usamos un Input layer explícito
modelo.add(Dense(1, activation='linear'))

# Definición del método de optimización (gradiente descendiente)
sgd = SGD(learning_rate=0.0004)
modelo.compile(loss='mse', optimizer=sgd)

# Imprimir en pantalla la información del modelo
modelo.summary()

# Entrenamiento: realizar la regresión lineal
num_epochs = 4000  # Número de iteraciones para el entrenamiento
batch_size = x.shape[0]  # Todos los datos se usarán en cada iteración
history = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Visualizar el proceso de entrenamiento
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.title('Pérdida durante el entrenamiento')
plt.show()

print("Entrenamiento finalizado")
# print(modelo.layers)
# Imprimir los coeficientes "w" y "b"
capas = modelo.layers[0]  # Cambia el índice si cambian las capas
w, b = capas.get_weights()
print(f'Parámetros: w = {w[0][0]:.4f}, b = {b[0]:.4f}')  # Mostrar valores de w y b con 4 decimales

# Graficar los datos originales y la regresión
y_regr = modelo.predict(x)
plt.figure(figsize=(12, 6))
plt.scatter(x, y, label='Datos originales')
plt.plot(x, y_regr, 'r', label='Regresión lineal')
plt.xlabel('Altura (m)')
plt.ylabel('Peso (kg)')
plt.title('Datos originales y regresión lineal')
plt.legend()
plt.show()

# Predicción
x_pred = np.array([1.80])  # Predicción para una altura de 1.80 metros
y_pred = modelo.predict(x_pred)
print("El peso será de {:.1f} kg para una altura de {} m".format(y_pred[0][0] * 100, x_pred[0]))
