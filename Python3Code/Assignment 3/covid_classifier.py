import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ConvLSTM2D, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.layers import Flatten
from keras.utils import np_utils


def split_sequence(sequence, targets, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps

        if end_ix > len(sequence) - 1:
            break

        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], targets[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


files = [os.path.join("./user_data/", file) for file in os.listdir("./user_data/")]
data = pd.concat((pd.read_csv(f) for f in files if (f.endswith('.csv') and not f.endswith('corrs.csv'))),
                 ignore_index=True).iloc[:, 1:]

enc = LabelEncoder()
enc_targets = enc.fit_transform(data.covid_symptoms_score.to_numpy())
targets = np_utils.to_categorical(enc_targets)

data.drop(['user_code', 'covid_symptoms_score'], inplace=True, axis=1)
features = data.values.astype('float32')

train_split = int(len(data) * .8)
train_X, train_y = features[:train_split], targets[:train_split]
test_X, test_y = features[train_split:], targets[train_split:]

EPOCHS = 500
BATCH_SIZE = 32
model = Sequential()


def convlstm():
    global train_X, train_y, test_X, test_y
    n_timesteps = 32
    train_X, train_y = split_sequence(train_X, train_y, n_timesteps)
    test_X, test_y = split_sequence(test_X, test_y, n_timesteps)

    n_features = train_X.shape[2]
    n_seq = 4
    n_steps = 8
    train_X = train_X.reshape(
        (train_X.shape[0], n_seq, 1, n_steps, n_features))  # Reshape into [samples, timesteps, features]
    test_X = test_X.reshape((test_X.shape[0], n_seq, 1, n_steps, n_features))

    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu',
                         input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(Flatten())


def lstm():
    # Werkt niet
    model.add(LSTM(100, activation='relu', input_length=train_X.shape[2]))
    model.add(Dropout(.2))
    model.add(Dense(50))


convlstm()
model.add(Dense(6, activation='softmax'))

es_monitor = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', tfa.metrics.FBetaScore(num_classes=6, average='micro', threshold=.5)])

history = model.fit(train_X, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(test_X, test_y), verbose=1, shuffle=False,
                    callbacks=[es_monitor, checkpoint])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title(f"Loss (categorical crossentropy)")
plt.legend()
plt.show()

test_results = model.predict(test_X, batch_size=BATCH_SIZE, verbose=2).flatten()
plt.plot(test_results, label='prediction', alpha=0.5)
plt.plot(test_y, label='actual')
plt.title(f"Evaluation test set actual vs. predicted covid score")
plt.legend()
plt.show()
