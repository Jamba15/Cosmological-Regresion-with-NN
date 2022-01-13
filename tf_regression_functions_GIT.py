import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from tensorflow.keras.layers import Dense, InputLayer
import pandas as pd
import os
from os.path import join
import fnmatch
import pickle as pk


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(join(root, name))
    return result


def integral(y, dx):
    return ((y[0] + y[-1]) / 2 + tf.reduce_sum(y[1:-1])) * dx


def bootstrap_sample(x, y, dy):
    new_x = np.zeros(x.shape)
    new_y = np.zeros(y.shape)
    new_dy = np.zeros(dy.shape)

    for i in range(0, x.shape[0]):
        index = np.random.randint(0, x.shape[0])
        new_x[i] = x[index]
        new_y[i] = y[index]
        new_dy[i] = dy[index]
    return new_x, new_y, new_dy


def load_dataset(filename, bootstrap, current_dir):
    """
    Load dataset from searching the filename from the folder given
    :param current_dir: Directory to start looking from
    :param bootstrap: Use or not the bootstrapping
    :param filename: filename of csv  dataset
    :return: 3 'tensors' the z domain, y, dy^2 and (x0, x0 domains) as a test
    """

    f_name = find(filename, current_dir)
    _, extent = os.path.splitext(f_name[0])

    if extent == '.txt':
        Data = pd.read_csv(f_name[0], usecols=[0, 1, 2], header=None, sep="   ", engine='python')
    elif extent == '.csv':
        Data = pd.read_csv(f_name[0], header=0, sep=';', engine='python')

    Data = Data.to_numpy(dtype='float32')
    x0 = Data[:, 0]
    y0 = Data[:, 1]
    dy0 = Data[:, 2]

    if bootstrap:
        x, y, dy = bootstrap_sample(x0, y0, dy0)
    else:
        x, y, dy = x0, y0, dy0

    H0 = 70E-6
    y = y / 5 + 1  # Ho y = log10(d) in pc
    y = y - np.log10(299792. / H0) - np.log10(1 + x)
    y = 10 ** y
    dy = y * dy

    x_data = x.copy()
    y_data = tf.constant(y, dtype=tf.float32)
    dy_data = tf.constant(dy, dtype=tf.float32)
    dy2_data = tf.math.pow(dy_data, 2)

    dx = 5.3E-4
    domains = [tf.range(0, x_data[i], dx, dtype=tf.float32) for i in range(x_data.shape[0])]
    M = max([domains[i].shape[0] for i in range(len(domains))])
    x_domain = tf.stack([tf.concat([domains[k], tf.zeros(M - domains[k].shape[0], dtype=tf.float32)], axis=0) for k in
                         range(len(domains))])

    domains = [tf.range(0, x0[i], dx, dtype=tf.float32) for i in range(x0.shape[0])]
    M = max([domains[i].shape[0] for i in range(len(domains))])
    x0_domain = tf.stack([tf.concat([domains[k], tf.zeros(M - domains[k].shape[0], dtype=tf.float32)], axis=0) for k in
                          range(len(domains))])

    return x_domain, y_data, (1 / dy2_data) / tf.reduce_sum(1 / dy2_data), x0, x0_domain


class Regress_Net(tf.keras.Model):
    def __init__(self,
                 hidden_size=20,
                 activation='tanh',
                 dx=5.3E-4,
                 omega_m=None):
        super(Regress_Net, self).__init__()

        self.Int_NN = tf.keras.Sequential()

        self.Int_NN.add(InputLayer(input_shape=1))

        for i in range(5):
            self.Int_NN.add(Dense(units=hidden_size,
                                  bias_initializer='zeros',
                                  activation=activation,
                                  use_bias=False))

        self.Int_NN.add(Dense(units=1,
                              activation=None,
                              bias_initializer='zeros',
                              use_bias=False))

        self.Int_NN.build()
        self.dx = tf.constant(dx, dtype=tf.float32)
        self.Om = tf.constant(omega_m, dtype=tf.float32)

    def call(self, inputs):
        b_size = tf.shape(inputs)[0]
        int_pred = tf.reshape(self.Int_NN(tf.reshape(inputs, [-1, 1])), [b_size, -1])
        to_sum = tf.pow(self.Om * tf.pow((1 + inputs), 3) + (1 - self.Om) * tf.exp(3 * int_pred), -0.5)

        return tf.reduce_sum(tf.where(inputs != 0, to_sum, 0), axis=1) * self.dx


def omega_training(config, save_ris=True, bootstrap=True, delete_last=False):
    om_str = str(config['omega_m'])
    current_dir = config['current_dir']
    mod = Regress_Net(omega_m=config['omega_m'])
    n_epochs = config['epochs']

    x_data, y_data, pesi_data, x0_test, x0_domain_test = load_dataset(filename=config['dataset'],
                                                                      bootstrap=bootstrap,
                                                                      current_dir=current_dir)

    mod.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=config['learning_rate']),
                loss='mean_squared_error',
                metrics=['mean_squared_error'],
                run_eagerly=False)

    patience = 35
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
                                                  min_delta=1E-7,
                                                  mode='min',
                                                  patience=patience)

    h = mod.fit(x_data,
                y_data,
                epochs=n_epochs,
                callbacks=stop_early,
                sample_weight=pesi_data,
                batch_size=100,
                verbose=2)
    current_dir = join(config['current_dir'], 'omega_m')

    if save_ris:
        name = os.path.splitext(config['dataset'])[0]

        # Saving integral
        dir = join(current_dir, 'Integral')
        os.makedirs(dir, exist_ok=True)
        integral = mod.Int_NN(x0_test.reshape([x0_test.shape[0], 1])).numpy()
        int_name = join(dir, name + om_str + '_integral.p')

        if os.path.exists(int_name) & delete_last:
            os.remove(int_name)

        with open(int_name, "ab") as f:
            pk.dump(integral, f)

        # Saving prediction
        dir = join(current_dir, 'Prediction')
        os.makedirs(dir, exist_ok=True)

        data_pred = mod.predict(x0_domain_test)
        pred_name = join(dir, name + om_str + '_prediction.p')

        if os.path.exists(pred_name) & delete_last:
            os.remove(pred_name)

        with open(pred_name, "ab") as f:
            pk.dump(data_pred, f)

        # Saving loss
        dir = join(current_dir, 'Loss')
        os.makedirs(dir, exist_ok=True)

        last_loss = np.array(h.history['mean_squared_error'][-1])
        loss_name = join(dir, name + om_str + '_loss.p')

        if os.path.exists(loss_name) & delete_last:
            os.remove(loss_name)

        with open(loss_name, "ab") as f:
            pk.dump(last_loss, f)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    x_data, y_data, pesi_data, x0_test, x0_domain_test = load_dataset(filename='simA.csv',
                                                                      bootstrap=False,
                                                                      current_dir=current_dir)
    mod = Regress_Net(omega_m=0.3)
    mod.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4),
                loss='mean_squared_error',
                metrics=['mean_squared_error'],
                run_eagerly=False)

    patience = 35
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
                                                  min_delta=1E-7,
                                                  mode='min',
                                                  patience=patience)

    h = mod.fit(x_data,
                y_data,
                epochs=200,
                callbacks=stop_early,
                sample_weight=pesi_data,
                batch_size=100,
                verbose=2)

    # mse = tf.keras.losses.MeanSquaredError()
    # ris = mse(y[1048:], mod.predict(x[1048:, :]), sample_weight=[peso[1048:]])

