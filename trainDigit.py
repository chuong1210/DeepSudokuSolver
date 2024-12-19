

# import os
# import cv2
# import numpy as np
# import pickle
# from sklearn.utils import shuffle
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import MaxPool2D, Conv2D, Flatten, Dense

# # 1. Hàm đọc ảnh từ folder Repository/digit_images
# import os
# import cv2
# import numpy as np

# def load_custom_digits(data_dir):
#     images = []
#     labels = []

#     # Đọc các thư mục từ sample_002 đến sample_010 (tương ứng với nhãn từ 1 đến 9)
#     for label, folder_name in enumerate([f"Sample{str(i).zfill(3)}" for i in range(2, 11)], start=1):
#         folder_path = os.path.join(data_dir, folder_name)
#         print(f"Đang kiểm tra folder: {folder_path}")

#         if not os.path.exists(folder_path):
#             print(f"Folder không tồn tại: {folder_path}")
#             continue  # Bỏ qua nếu folder không tồn tại

#         for img_name in os.listdir(folder_path):
#             img_path = os.path.join(folder_path, img_name)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 print(f"Lỗi đọc ảnh: {img_path}")
#                 continue
#             img_resized = cv2.resize(img, (28, 28))  # Resize ảnh về 28x28
#             images.append(img_resized)
#             labels.append(label)

#     images = np.array(images)
#     labels = np.array(labels)

#     return images, labels

# # Sử dụng đường dẫn tuyệt đối


# # 2. Load dataset Sudoku và MNIST
# def load_datasets():
#     # Load dataset từ file pickle
#     with open("Repository/digitDataset.pkl", "rb") as file:
#         dataset = pickle.load(file)

#     x1, y1 = dataset['data'], dataset['target']
#     y1 = y1.reshape(len(y1),)

#     # Shuffle dataset Sudoku
#     x1, y1 = shuffle(x1, y1, random_state=0)

#     # Chia train-test cho dataset Sudoku
#     x_train_1, x_test_1, y_train_1, y_test_1 = x1[:1048], x1[1048:], y1[:1048], y1[1048:]

#     # Load dataset MNIST
#     (x_train_2, y_train_2), (x_test_2, y_test_2) = mnist.load_data()

#     # Gộp dataset Sudoku và MNIST
#     x_train = np.concatenate((x_train_1, x_train_2), axis=0)
#     x_test = np.concatenate((x_test_1, x_test_2), axis=0)
#     y_train = np.concatenate((y_train_1, y_train_2), axis=0)
#     y_test = np.concatenate((y_test_1, y_test_2), axis=0)

#     return x_train, y_train, x_test, y_test

# # 3. Tiền xử lý dữ liệu
# def preprocess_data(x_train, y_train, x_test, y_test, custom_images, custom_labels):
#     # Gộp dữ liệu custom vào dataset huấn luyện
#     x_train = np.concatenate((x_train, custom_images), axis=0)
#     y_train = np.concatenate((y_train, custom_labels), axis=0)

#     # Reshape dữ liệu về (28, 28, 1) cho CNN
#     x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#     x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#     # Chuẩn hóa dữ liệu về [0, 1]
#     x_train = x_train.astype('float32') / 255.0
#     x_test = x_test.astype('float32') / 255.0

#     # One-hot encode nhãn
#     y_train_OHE = to_categorical(y_train)
#     y_test_OHE = to_categorical(y_test)

#     # Shuffle dữ liệu huấn luyện
#     x_train, y_train_OHE = shuffle(x_train, y_train_OHE, random_state=42)

#     return x_train, y_train_OHE, x_test, y_test_OHE

# # 4. Xây dựng model CNN
# def build_model():
#     model = Sequential()

#     # Thêm các layer
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#     model.add(Conv2D(32, (3, 3), activation='relu'))
#     model.add(MaxPool2D())

#     model.add(Conv2D(16, (3, 3), activation='relu', padding='SAME'))
#     model.add(Conv2D(64, (5, 5), activation='relu', padding='SAME'))
#     model.add(Conv2D(32, (1, 1), activation='relu', padding='SAME'))
#     model.add(MaxPool2D())

#     model.add(Conv2D(8, (3, 3), activation='relu', padding='SAME'))
#     model.add(Conv2D(32, (5, 5), activation='relu', padding='SAME'))
#     model.add(Conv2D(16, (1, 1), activation='relu', padding='SAME'))
#     model.add(Flatten())

#     model.add(Dense(400, activation='relu'))
#     model.add(Dense(10, activation='softmax'))

#     # Compile model
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     return model

# # 5. Huấn luyện và lưu model
# def train_and_save_model():
#     # Load dữ liệu
#     x_train, y_train, x_test, y_test = load_datasets()
#     data_dir = r"c:\Users\chuon\Desktop\DoAnTNTT\Repository\digit_images"
#     custom_images, custom_labels = load_custom_digits(data_dir)

    
#     # Tiền xử lý dữ liệu
#     x_train, y_train_OHE, x_test, y_test_OHE = preprocess_data(x_train, y_train, x_test, y_test, custom_images, custom_labels)
    
#     # Xây dựng model
#     model = build_model()
    
#     # Huấn luyện model
#     model.fit(x_train, y_train_OHE, validation_data=(x_test, y_test_OHE), epochs=15)
    
#     # Lưu model
#     model.save("model_sudoku_mnist_custom.keras", overwrite=True)

# # Gọi hàm để huấn luyện và lưu model
# train_and_save_model()


import numpy as np
import sklearn
import tensorflow
import cv2

import pickle


def train_and_save_model():
    '''Lưu mô hình CNN được đào tạo trên một tập dữ liệu bao gồm các chữ số lấy từ tạp chí sudoku'''

    file = open("Repository/digitDataset.pkl", "rb")
    dataset = pickle.load(file)
    file.close()

    x, y = dataset['data'], dataset['target']
    y = y.reshape(1310,)

    # shuffle the dataset
    from sklearn.utils import shuffle
    x, y = shuffle(x, y, random_state=0)

    # train-test splitting
    x_train, x_test, y_train, y_test = x[:1048], x[1048:], y[:1048], y[1048:]

    # reshape data to fit model
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # normalize the train/test dataset
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    from tensorflow.keras.utils import to_categorical
    # one-hot encode target column
    y_train_OHE = to_categorical(y_train)
    y_test_OHE = to_categorical(y_test)


    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import MaxPool2D, Conv2D, Flatten, Dense

    # create the model
    model = Sequential()

    # add model layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(16, (3, 3), activation='relu', padding='SAME'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='SAME'))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='SAME'))
    model.add(MaxPool2D())

    model.add(Conv2D(8, (3, 3), activation='relu', padding='SAME'))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='SAME'))
    model.add(Conv2D(16, (1, 1), activation='relu', padding='SAME'))
    model.add(Flatten())

    model.add(Dense(400, activation='relu'))
    model.add(Dense(10, activation='softmax'))
# Sử dụng Adam (một phương pháp tối ưu hóa phổ biến) để điều chỉnh trọng số trong quá trình huấn luyện.

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train_OHE, validation_data=(x_test, y_test_OHE), epochs=15)

    # save the model
    model.save("models/model_sudoku.keras", overwrite=True)

train_and_save_model()