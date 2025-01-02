import tensorflow as tf

# Tải mô hình TensorFlow đã huấn luyện (classification_model.h5)
model = tf.keras.models.load_model(r'C:\Users\admin\PycharmProjects\Diemdanh\tiensuly\checkpoints\classification_model.h5')

# Chuyển đổi mô hình sang TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Lưu mô hình TensorFlow Lite vào file .tflite
tflite_model_path = r'C:\Users\admin\PycharmProjects\Diemdanh\tiensuly\checkpoints\classification_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model converted to TFLite and saved at: {tflite_model_path}")
