import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib  # Để lưu mô hình đã huấn luyện

# Tải dữ liệu chữ số MNIST
digits = datasets.load_digits()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (min-max scaling)
x_train = x_train / 16.0
x_test = x_test / 16.0

# Huấn luyện mô hình KNN với k=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(x_train, y_train)

# Lưu mô hình đã huấn luyện
joblib.dump(knn_classifier, 'models/knn_digits_model.pkl')
