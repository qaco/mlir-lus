from keraslus import layers
from keraslus import keras

def model():
  input_audio = keras.Input(shape=(49,40), batch_size=1)
  net = layers.LSTM2(units=4,
                    activation='tanh',
                    recurrent_activation='softmax')(input_audio)
  net = layers.Flatten()(net)
  net = layers.Dense(units=4, activation='relu')(net)
  net = layers.Dense(units=4, activation='relu')(net)
  net = layers.Dense(units=4)(net)
  return keras.Model(input_audio, net)

model = model()
print(model)
