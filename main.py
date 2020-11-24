from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

# cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

# y estos son los resultados que se obtienen, en el mismo orden
target_data = np.array([[0], [1], [1], [0]], "float32")

# se crea el modelo
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

# se entrena el modelo
model.fit(training_data, target_data, epochs=1000)

# se evalua el modelo
scores = model.evaluate(training_data, target_data)

# se imprimen las metricas y predicion
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print(model.predict(training_data).round())
