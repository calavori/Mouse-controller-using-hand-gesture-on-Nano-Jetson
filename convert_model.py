import tensorflow as tf

model_path = 'models'
model = tf.keras.models.load_model(model_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)
