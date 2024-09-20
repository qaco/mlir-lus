from keraslus import layers
from keraslus import keras

def model():
  inp = keras.Input(shape=(5,3,1),time_major=True)
  x = layers.LSTM(units=100,
                    activation='relu',
                    recurrent_activation='sigmoid')(inp)
  x = layers.Dense(units=50, activation="relu")(x)
  x = layers.Dense(units=50,activation="relu")(x)
  x = layers.Dense(units=1,activation=None)(x)
  return keras.Model(inp, x)

model = model()
print(model)
