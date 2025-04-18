import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import pickle as pkl
import argparse
import sys
import os
import tqdm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input, Dense
from tensorflow.keras.models import load_model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


@tf.keras.utils.register_keras_serializable()
class ForwardNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ForwardNet, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(2)

    def call(self, x):
        h = self.dense1(x)
        h = self.dense2(h)
        h = self.dense3(h)
        return self.out(h)

    def get_config(self):
        config = super(ForwardNet, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
def train_forward_model(forward_net, args):
    X, y = df.iloc[:, 0:3], df.iloc[:, 3:5]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=args.seed)

    forward_net.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                        loss='mse',
                        metrics=['mse'])

    model_path = ''.format(args.eval)
    log_path = '/forward_model_{}_eval{}_log.pkl'.format('MLP', args.eval)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=args.epochs // 5, verbose=1)

    history = forward_net.fit(X_train, y_train,
                              epochs=args.epochs,
                              batch_size=64,
                              validation_data=(X_val, y_val),
                              callbacks=[checkpoint, reduce_lr])

    forward_training_log = {'train_loss': history.history['loss'],
                            'val_mse': history.history['val_loss']}

    with open(log_path, 'wb') as f:
        pkl.dump(forward_training_log, f)

    forward_net.load_weights(model_path)
    test_loss = forward_net.evaluate(X_test, y_test, batch_size=128)
    print(f'test loss {test_loss[0]}')

    forward_training_log['test_mse'] = test_loss[0]

    with open(log_path, 'wb') as f:
        pkl.dump(forward_training_log, f)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='train a separate model for evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)

    if 'google.colab' in sys.modules or 'ipykernel' in sys.modules:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    forward_net = ForwardNet()
    train_forward_model(forward_net, args)