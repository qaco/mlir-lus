from tensorflow.python.keras import layers
from tensorflow.python import keras
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
tf1.reset_default_graph()
config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf1.Session(config=config)
tf1.keras.backend.set_session(sess)

# def MyNetwork():
#     input_shape=(8,8,5)
#     inp = keras.Input(shape=input_shape)
#     x = layers.LayerNormalization(
#         axis=[2,3],
#         epsilon=0.001,
#         center=True,
#         scale=True,
#         beta_initializer="zeros",
#         gamma_initializer="ones",
#         beta_regularizer=None,
#         gamma_regularizer=None,
#         beta_constraint=None,
#         gamma_constraint=None,
#     )(inp)
#     model = keras.Model(inp, x, name="mynet")
#     return model

# def MyNetwork(input_shape=(3,1)):
#     inp = keras.Input(shape=input_shape)
#     x = layers.LSTM(50, activation='relu', recurrent_activation="sigmoid", name='lstm')(inp)
#     # x = layers.Dense(1, name='dense')(x)
#     model = keras.Model(inp, x, name="rnn")
#     return model

def MyNetwork(input_shape=(1,1)):
  inp = keras.Input(shape=(1,1), batch_size=3)
  x = layers.LSTM(units=100,
                    activation='relu',
                    recurrent_activation='sigmoid',
                  time_major=True)(inp)
  x = layers.Dense(units=50, activation="relu")(x)
  x = layers.Dense(units=50,activation="relu")(x)
  x = layers.Dense(units=1,activation=None)(x)
  return keras.Model(inp, x)

model = MyNetwork()

print([v.name for v in model.inputs])
print([v.shape for v in model.inputs])
print([v.name for v in model.outputs])
print([v.shape for v in model.outputs])

tf1.train.write_graph(sess.graph_def, '.', 'protobuf.pb')
