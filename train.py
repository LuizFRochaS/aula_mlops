import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import git
import random
import mlflow
import numpy as np

def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

def set_mlflow_tracking_uri(username, password, uri):
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = password
    mlflow.set_tracking_uri(uri)

def load_data(url):
    data = pd.read_csv(url)
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)
    return X_df, y

def preprocess_labels(y_train, y_test):
    y_train = y_train - 1
    y_test = y_test - 1
    return y_train, y_test

def build_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(10, activation='relu' ))
    model.add(Dense(10, activation='relu' ))
    model.add(Dense(3, activation='softmax' ))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=50, validation_split=0.2, verbose=3):
    with mlflow.start_run(run_name='train_experiment') as run:
        model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=verbose)
    return model

# Configuração do MLflow
MLFLOW_TRACKING_USERNAME = 'luizfrs.it'
MLFLOW_TRACKING_PASSWORD = '839c9880f528ea437ad44ed37dc48365f765c235'
MLFLOW_TRACKING_URI = 'https://dagshub.com/luizfrs.it/mlops_18-05-23.mlflow'

# Configuração do ambiente
reset_seeds()
set_mlflow_tracking_uri(MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_URI)

# Carregamento dos dados
data_url = 'https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv'
X, y = load_data(data_url)

# Pré-processamento das etiquetas
y_train, y_test = preprocess_labels(y_train, y_test)

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Construção do modelo
model = build_model(input_shape=(X_train.shape[1],))

# Treinamento do modelo
trained_model = train_model(model, X_train, y_train)

#--------------------older--------------------------------------

# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, InputLayer

# import pandas as pd
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split

# import os
# import git
# import random
# import mlflow
# import numpy as np


# def reset_seeds():
#     os.environ['PYTHONHASHSEED'] = str(42)
#     tf.random.set_seed(42)
#     np.random.seed(42)
#     random.seed(42)


# MLFLOW_TRACKING_URI = 'https://dagshub.com/luizfrs.it/mlops_18-05-23.mlflow'
# MLFLOW_TRACKING_USERNAME = 'luizfrs.it'
# MLFLOW_TRACKING_PASSWORD = '839c9880f528ea437ad44ed37dc48365f765c235'

# os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
# os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# mlflow.tensorflow.autolog(log_models=True,
#                           log_input_examples=True,
#                           log_model_signatures=True)

# data = pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')
# X = data.drop(["fetal_health"], axis=1)
# y = data["fetal_health"]

# columns_names = list(X.columns)
# scaler = preprocessing.StandardScaler()
# X_df = scaler.fit_transform(X)
# X_df = pd.DataFrame(X_df, columns=columns_names)

# X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

# y_train = y_train -1
# y_test = y_test - 1

# reset_seeds()
# model = Sequential()
# model.add(InputLayer(input_shape=(X_train.shape[1], )))
# model.add(Dense(10, activation='relu' ))
# model.add(Dense(10, activation='relu' ))
# model.add(Dense(3, activation='softmax' ))

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# with mlflow.start_run(run_name='train_experiment') as run:
#     model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=3)
