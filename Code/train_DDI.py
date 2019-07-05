
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
from load_data import load_data
from eval_DDI import evaluate

from keras import backend as K
import tensorflow as tf

n_input = 100
n_output = 86

n_epods = 100
batch_size = 256
learning_rate = 0.0001

R_result = np.zeros((10, 7))
if __name__ == "__main__":

    for num_CV in range(0, 10):
        print('num_CV...............', num_CV)
        model = Sequential()
        model.add(Dense(2048, kernel_initializer="uniform", input_shape=(256,), activation="relu"))  # input layer
        for i in range(8):
            model.add(Dense(2048, kernel_initializer="uniform", input_shape=(2048,), activation='relu'))      # 9 hidden layers
        model.add(Dense(n_output, kernel_initializer="uniform", input_shape=(2048,), activation='sigmoid'))    # output layer

        adma = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=adma, loss='binary_crossentropy')

        train_x, train_y, val_x, val_y, test_x, test_y = load_data(num_CV)
        test_num = test_x.shape[0]
        print('test_num.............', test_num)
        val_data = (val_x, val_y)

        model.fit(train_x, train_y, validation_data=val_data, epochs=n_epods, batch_size=batch_size)
        model.save('model_weight.h5')
        pred = model.predict(test_x)
        ma, ma_pre, ma_rea, mi, mi_pre, mi_rea, m_acc = evaluate(pred, test_y, num_CV)

        R_result[num_CV, 0] = ma
        R_result[num_CV, 1] = ma_pre
        R_result[num_CV, 2] = ma_rea
        R_result[num_CV, 3] = mi
        R_result[num_CV, 4] = mi_pre
        R_result[num_CV, 5] = mi_rea
        R_result[num_CV, 6] = m_acc

    np.savetxt("data/output/predict_values.txt", R_result)

