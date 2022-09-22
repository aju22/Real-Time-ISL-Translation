import os
from utils import accuracy, get_data
import tensorflow as tf
from models import load_model


model = load_model('lstm_v3')
print(model.summary())

X_test, y_test = get_data(train=False)

accuracy_score = accuracy(model, X_test, y_test)
print(f"\nAccuracy : {accuracy_score}")



