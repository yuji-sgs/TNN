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
class InverseNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(InverseNet, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu', input_shape=(2,))
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(3)

    def call(self, y):
        x = self.dense1(y)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.out(x)
        return x

    def get_config(self):
        config = super(InverseNet, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class TandemNet(tf.keras.Model):
    def __init__(self, forward_model, inverse_model, **kwargs):
        super(TandemNet, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def call(self, y):
        x_pred = self.inverse_model(y)
        y_pred = self.forward_model(x_pred)
        return x_pred, y_pred

    def get_config(self):
        config = {
            'forward_model': self.forward_model.get_config(),
            'inverse_model': self.inverse_model.get_config(),
        }
        base_config = super(TandemNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        forward_model = ForwardNet.from_config(config['forward_model'])
        inverse_model = InverseNet.from_config(config['inverse_model'])
        return cls(forward_model, inverse_model)
    
    
def train_inverse_model(epochs, device):
    X, y = df.iloc[:, 0:3], df.iloc[:, 3:5]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=args.seed)

    forward_model = ForwardNet()
    forward_model = load_model('')
    forward_model.trainable = False

    inverse_model = InverseNet()
    tandem_net = TandemNet(forward_model, inverse_model)

    optimizer = optimizers.Adam(learning_rate=5e-4)
    loss_fn = losses.MeanSquaredError()

    train_dataset = tf.data.Dataset.from_tensor_slices((y_train.values, (X_train.values, y_train.values)))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(256)

    val_dataset = tf.data.Dataset.from_tensor_slices((y_val.values, (X_val.values, y_val.values)))
    val_dataset = val_dataset.batch(128)

    best_val_loss = float('inf')

    lambda_y = 1000

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for step, (y_batch_train, (x_batch_target, y_batch_target)) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x_pred, y_pred = tandem_net(y_batch_train)
                loss_x = loss_fn(x_batch_target, x_pred)
                loss_y = loss_fn(y_batch_target, y_pred)
                loss = loss_x + lambda_y * loss_y
            grads = tape.gradient(loss, inverse_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, inverse_model.trainable_weights))

        val_losses_x = []
        val_losses_y = []
        for y_batch_val, (x_batch_val_target, y_batch_val_target) in val_dataset:
            x_pred, y_pred = tandem_net(y_batch_val)
            val_loss_x = loss_fn(x_batch_val_target, x_pred)
            val_loss_y = loss_fn(y_batch_val_target, y_pred)
            val_losses_x.append(val_loss_x.numpy())
            val_losses_y.append(val_loss_y.numpy())

        mean_val_loss_x = np.mean(val_losses_x)
        mean_val_loss_y = np.mean(val_losses_y)
        mean_val_loss = mean_val_loss_x + lambda_y * mean_val_loss_y

        print(f'Validation Loss (x): {mean_val_loss_x}')
        print(f'Validation Loss (y): {mean_val_loss_y}')
        print(f'Validation Loss (total): {mean_val_loss}')

        if mean_val_loss_y < best_val_loss:
            inverse_model.save('')
            best_val_loss = mean_val_loss_y
            print('###Serializing model...')
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    args, unknown = parser.parse_known_args()

    seed = args.seed
    epochs = args.epochs

    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_inverse_model(epochs, args.device)