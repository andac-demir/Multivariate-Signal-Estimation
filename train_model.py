from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, GRU
 
'''
    Dataset is a 2D numpy array where the columns represent the fetures 
    (variables) and rows represent each time step t-2, t-1, t, t+1, t+2, etc.
    n_in is the number of samples to go back in the past and
    n_out is the number of samples to predict in the future at each time step
    We predict the EEG at the current time step
    Supervised data is a 2D array. 
    Rows show the time steps and columns show the features
    Features are past EEG data of one selected channel (out of 15), 
    4 EMG channels and force measurement (z-dimension) and
    surface category (digital trigger) in this order, 7 features in total  
    There are n_features * n_in number of columns in total.
'''
def series_to_supervised(data, n_in, n_out, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data) 
	cols, names = list(), list()
	# input sequence (t-n_in, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n_out)
	for i in range(n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
def preprocessing(filename, n_in, n_out):
    # TODO: convert the numpy arrays to a CSV file
    dataset = read_csv(filename, header=0, index_col=0)
    values = dataset.values
    # ensures all data is float
    values = values.astype('float32')
    # feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, n_out)
    print(reframed.shape)
    return scaler, reframed
 
def train_test_split(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_samples = int(values.shape[0] * 0.8)
    train = values[:n_train_samples, :]
    test = values[n_train_samples:, :]
    return train, test

# splits train and test sets into input and outputs
def input_output_split(train, test, n_in, n_features):
    n_obs = n_in * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y
 
# design network and train the network
def train_network(train_X, train_y, test_X, test_y):
    model = Sequential()
    # input shape that is passed into the network: [timesteps, features]
    model.add(GRU(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=64, 
                        validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    return model
 
def main():
    # specify the number of lag samples, n_out must be always 1
    n_in, n_out = 3, 1
    n_features = 7 
    scaler, reframed = preprocessing(filename, n_in, n_out)
    train, test = train_test_split(reframed)
    train_X, train_y, test_X, test_y = input_output_split(train, test, 
                                                          n_in, n_features)
    model = train_network(train_X, train_y, test_X, test_y)
    # make a prediction
    yhat = model.predict(test_X)
    # reshape test input from 3D back to 2D
    test_X = test_X.reshape((test_X.shape[0], n_in*n_features))
    # inverse scaling for forecast using the scaled EEG data of past time steps
    inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # inverse scaling for actual EEG
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    pyplot.plot(inv_yhat)
    pyplot.plot(inv_y)
    pyplot.show()
    # To see the results more clearly
    pyplot.plot(inv_yhat[-100:])
    pyplot.plot(inv_y[-100:])
    pyplot.show()


if __name__ == "__main__":
    main()

