import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

model = load_model('../model.h5')

tfjs.converters.save_keras_model(model, '../')
