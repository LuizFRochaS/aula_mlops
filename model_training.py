import os
import random
import mlflow
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, InputLayer

from config import MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
from data_processing import load_data, preprocess_data

def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

def create_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.tensorflow.autolog(log_models=True,
                              log_input_examples=True,
                              log_model_signatures=True)

    with mlflow.start_run(run_name='train_experiment') as run:
        model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=3)

if __name__ == "__main__":
    reset_seeds()
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = create_model(input_shape=(X_train.shape[1],))
    train_model(model, X_train, y_train)
