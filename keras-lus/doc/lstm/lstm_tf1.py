from tensorflow.python.keras import layers
from tensorflow.python import keras
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
tf1.reset_default_graph()
config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf1.Session(config=config)
tf1.keras.backend.set_session(sess)

def MyNetwork(input_shape=(3,1)):
    inp = keras.Input(shape=input_shape)
    x = layers.LSTM(50, activation='relu', name='lstm')(inp)
    x = layers.Dense(1, name='dense')(x)
    model = keras.Model(inp, x, name="rnn")
    return model

model = MyNetwork()

print([v.name for v in model.inputs])
print([v.name for v in model.outputs])

tf1.train.write_graph(sess.graph_def, '.', 'protobuf.pb')
