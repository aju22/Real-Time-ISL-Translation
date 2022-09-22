import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from models import load_model

device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
IMPORT_PATH = os.path.join('keypoint_data')
actions = os.listdir(IMPORT_PATH)
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:

    for videofile in os.listdir(os.path.join(IMPORT_PATH, action)):
        data = np.load(os.path.join(IMPORT_PATH, action, videofile))
        sequences.append(data)
        labels.append(label_map[action])

X = np.array(sequences, dtype = np.float32)
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = load_model('lstm_v3', pretrained=False, device=device)
callbacks = [

    tf.keras.callbacks.ModelCheckpoint("isl_model.h5", save_best_only=True),

    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=10,
                                         verbose=0,
                                         mode='auto',
                                         min_delta=0.0001,
                                         cooldown=15,
                                         min_lr=0),

    # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
]

with tf.device(device):
    history = model.fit(X_train, y_train,
                        epochs=500,
                        batch_size=64,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)









