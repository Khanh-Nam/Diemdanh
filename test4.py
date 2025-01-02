import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from sklearn.metrics import roc_auc_score
import face_recognition

# Kiểm tra TensorFlow
print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")


# Hàm lấy embedding từ ảnh sử dụng face_recognition
def get_embedding(img_path):
    try:
        # Đọc ảnh
        img = face_recognition.load_image_file(img_path)

        # Lấy embeddings của khuôn mặt đầu tiên (nếu có)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            # Nếu có khuôn mặt, trả về embedding của khuôn mặt đầu tiên
            return encodings[0]
        else:
            print(f"No face found in {img_path}")
            return None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


# Hàm tải dataset từ thư mục
def load_dataset(data_dir):
    data = []
    labels = []
    label_encoder = LabelEncoder()

    for label in tqdm(os.listdir(data_dir), desc="Loading dataset"):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for image_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, image_file)
            embedding = get_embedding(img_path)
            if embedding is not None:
                data.append(embedding)
                labels.append(label)

    labels = label_encoder.fit_transform(labels)
    return np.array(data), np.array(labels), label_encoder


# Đường dẫn dữ liệu
data_dir = r"C:\Users\admin\Desktop\video\anhsinhvien"

# Tải toàn bộ dữ liệu và chia thành tập train và test (80% train, 20% test)
all_embeddings, all_labels, label_encoder = load_dataset(data_dir)
train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
    all_embeddings, all_labels, test_size=0.2, random_state=42
)


# Hàm xây dựng mô hình phân loại
def classification_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Dense(256, activation="relu")(inputs)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)


# Xây dựng và biên dịch mô hình
num_classes = len(np.unique(train_labels))
clf_model = classification_model((train_embeddings.shape[1],), num_classes)
clf_model.compile(optimizer=Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Huấn luyện mô hình
clf_model.fit(train_embeddings, train_labels, epochs=50, batch_size=32)

# Lưu mô hình và bộ mã hóa nhãn vào thư mục mới
checkpoints_dir = r"C:\Users\admin\PycharmProjects\Diemdanh\tiensuly\checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

# Lưu mô hình
clf_model.save(os.path.join(checkpoints_dir, "classification_model.h5"))

# Lưu bộ mã hóa nhãn
with open(os.path.join(checkpoints_dir, "label_encoder.pkl"), "wb") as file:
    pickle.dump(label_encoder, file)
print("Model and Label Encoder saved successfully.")

# Đánh giá mô hình trên tập test
test_loss, test_accuracy = clf_model.evaluate(test_embeddings, test_labels, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Tính toán AUC-ROC
y_pred_proba = clf_model.predict(test_embeddings)  # Xác suất cho từng lớp
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
auc_roc = roc_auc_score(test_labels_one_hot, y_pred_proba, multi_class='ovr', average='macro')
print(f"Test AUC-ROC: {auc_roc}")

# Chuyển mô hình sang TensorFlow Lite
tflite_model = tf.lite.TFLiteConverter.from_keras_model(clf_model)
tflite_model_path = "classification_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model.convert())
print(f"Model converted to TensorFlow Lite and saved at {tflite_model_path}")
