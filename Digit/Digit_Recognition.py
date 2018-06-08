import pandas as pd
from sklearn.model_selection import train_test_split
# from __future__ import division, print_function, absolute_import
import tflearn
import tflearn.data_utils as du
import numpy as np


def Get_data():
    print("get data")
    train_data = pd.read_csv('data/train.csv')
    predict_data = pd.read_csv('data/test.csv')
    train_X = train_data.drop('label', axis=1)
    train_Y = train_data[['label']]
    data = {'X_train': train_X,
            'Y_train': train_Y,
            'predict_data': predict_data}
    return data


def train(X_train_R, Y_train, X_test_R, Y_test, predict_data):
    X_train_R, mean = du.featurewise_zero_center(X_train_R)
    X_test_R = du.featurewise_zero_center(X_test_R, mean)

    net = tflearn.input_data(shape=[None, 28, 28, 1])
    net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)

    net = tflearn.residual_bottleneck(net, 3, 16, 64)
    net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 32, 128)
    net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 64, 256)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    net = tflearn.fully_connected(net, 10, activation='softmax')

    net = tflearn.regression(
        net,
        optimizer='momentum',
        loss='categorical_crossentropy',
        learning_rate=0.1
    )
    model = tflearn.DNN(
        net,
        checkpoint_path='model_Digit',
        max_checkpoints=2,
        tensorboard_verbose=0,
        tensorboard_dir='logs'
    )
    model.fit(
        X_train_R,
        Y_train,
        n_epoch=1,
        validation_set=(X_test_R, Y_test),
        show_metric=True,
        batch_size=10,
        run_id='Digit'
    )
    model.save("model_Digit/digit_recognition")
    print("model train OK!\n save model OK!")

    model.load("model_Digit/digit_recognition")
    print("model already loaded!")

    print("model predict")
    label = model.predict(predict_data)
    print("predict label: ", label)

if __name__ == '__main__':
    data = Get_data()
    X_train, X_test, Y_train, Y_test = train_test_split(data['X_train'], data['Y_train'], test_size=0.2)

    X_train_R = X_train.values.reshape([-1, 28, 28, 1])
    X_test_R = X_test.values.reshape([-1, 28, 28, 1])

    # X_train_R = X_train_R.values
    Y_train = Y_train.values

    # X_test_R = X_test_R.values
    Y_test = Y_test.values
    predict_test = data['predict_data'].values

    print(X_train_R.shape, Y_train.shape, X_test_R.shape, Y_test.shape, predict_test.shape)
    train(X_train_R, Y_train, X_test_R, Y_test, predict_test)
