import os
import gzip
import pickle
import numpy as np

os.chdir("data")
fn = os.listdir()[0]

def load_data():
    with gzip.open(fn, 'rb') as f:
        (training_data, validation_data, test_data) \
             = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    train_d, valid_d, test_d = load_data()
    traning_inputs = [np.reshape(picture, (28**2, 1)) \
        for picture in train_d[0]]
    training_results = [vectorize_result(answer) \
        for answer in train_d[1]]
    training_data = list(zip(traning_inputs, training_results))
    validation_inputs = [np.reshape(picture, (28**2, 1)) \
        for picture in valid_d[0]]
    validation_data = list(zip(validation_inputs, valid_d[1]))
    test_inputs = [np.reshape(picture, (28**2, 1)) \
        for picture in test_d[0]]
    test_data = list(zip(test_inputs, test_d[1]))
    return (training_data, validation_data, test_data)

def vectorize_result(answer):
    e = np.zeros((10, 1))
    e[answer] = 1.0
    return e
