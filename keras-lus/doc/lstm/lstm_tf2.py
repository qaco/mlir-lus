from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as tf
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features), name="lstm"))
model.add(Dense(1, name="dense"))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

@tf.function()
def predict_step_func(x, y):
    model.predict_step((x, y))

concrete_func = predict_step_func.get_concrete_function(
    x=tf.TensorSpec(shape=x_input.shape, dtype=tf.float32),
    y=tf.TensorSpec(shape=yhat.shape, dtype=tf.float32),
)

# Convert to MLIR
mlir = tf.mlir.experimental.convert_function(
    concrete_func, pass_pipeline="tf-standard-pipeline"
)

# Print
with open("lstm.mlir", "w") as f:
    f.write(mlir)

model.save_weights('lstm.h5')
