# import csv
# import random
# import uuid

# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# def generate_sudoku_data(num_rows=300):
#     """
#     Generates random Sudoku solving time data for a CSV file.

#     Args:
#         num_rows: The number of rows to generate.

#     Returns:
#         A list of lists, where each inner list represents a row of data.
#     """

#     data = [
#         ["Người chơi", "Thời gian giải dễ (phút)", "Thời gian giải trung bình (phút)", "Thời gian giải khó (phút)"]
#     ]

#     # Define difficulty ranges (adjust as needed)
#     easy_range = (1, 10)  # Example: 1 to 10 minutes for easy
#     medium_range = (5, 20)  # Example: 5 to 20 minutes for medium
#     hard_range = (10, 30)  # Example: 10 to 30 minutes for hard


#     for i in range(num_rows):
#         player_name = f"Người {i + 1}"  #  A more organized name format
#         easy_time = random.randint(*easy_range)
#         medium_time = random.randint(max(easy_time, medium_range[0]), medium_range[1])
#         hard_time = random.randint(max(medium_time, hard_range[0]), hard_range[1])

#         row = [player_name, easy_time, medium_time, hard_time]
#         data.append(row)

#     return data



# def write_data_to_csv(data, filename="sudoku_times.csv"):
#     """Writes the generated data to a CSV file."""
#     with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(data)

# # if __name__ == "__main__":
# #     generated_data = generate_sudoku_data(300)
# #     write_data_to_csv(generated_data)
# #     print("Dữ liệu đã được tạo và lưu vào sudoku_times.csv")
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# # 1. Chuẩn bị dữ liệu: Giả sử chúng ta có bảng điểm của 300 người chơi
# # Mỗi người chơi sẽ có thời gian giải Sudoku cho 3 mức độ (dễ, trung bình, khó)

# # Tạo dữ liệu mẫu giả
# np.random.seed(42)  # Để tái tạo lại dữ liệu cho lần sau
# num_players = 300

# # Giả lập thời gian giải của các người chơi (giới hạn thời gian giải trong phút)
# easy_times = np.random.randint(1, 20, size=num_players)  # Thời gian giải cho mức độ dễ (1-20 phút)
# medium_times = np.random.randint(5, 30, size=num_players)  # Thời gian giải cho mức độ trung bình (5-30 phút)
# hard_times = np.random.randint(10, 40, size=num_players)  # Thời gian giải cho mức độ khó (10-40 phút)

# # Tạo bảng dữ liệu
# data = pd.DataFrame({
#     'easy': easy_times,
#     'medium': medium_times,
#     'hard': hard_times
# })

# # 2. Tiến hành chuẩn hóa dữ liệu
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)

# # 3. Áp dụng K-means clustering
# # Giả sử chúng ta sẽ phân thành 3 nhóm: "Gà", "Decent", "Pro"
# kmeans = KMeans(n_clusters=3, random_state=42)
# data['cluster'] = kmeans.fit_predict(scaled_data)

# # 4. Đánh giá nhóm của người chơi
# # Chúng ta có thể gán các nhãn theo các nhóm: "Gà", "Decent", "Pro" dựa trên thời gian giải
# def assign_label(row, difficulty):
#     if difficulty == 'easy':
#         if row['easy'] > 10:
#             return 'Gà'
#         elif row['easy'] <= 5:
#             return 'Pro'
#         else:
#             return 'Decent'
#     elif difficulty == 'medium':
#         if row['medium'] > 15:
#             return 'Gà'
#         elif row['medium'] <= 6:
#             return 'Pro'
#         else:
#             return 'Decent'
#     else:  # 'hard'
#         if row['hard'] > 20:
#             return 'Gà'
#         elif row['hard'] <= 9:
#             return 'Pro'
#         else:
#             return 'Decent'

# # Áp dụng phân loại dựa trên các mức độ khó
# data['easy_label'] = data.apply(lambda row: assign_label(row, 'easy'), axis=1)
# data['medium_label'] = data.apply(lambda row: assign_label(row, 'medium'), axis=1)
# data['hard_label'] = data.apply(lambda row: assign_label(row, 'hard'), axis=1)

# # 5. Kiểm tra kết quả phân nhóm
# print(data[['easy', 'medium', 'hard', 'cluster', 'easy_label', 'medium_label', 'hard_label']].head())

# # 6. Visualize kết quả phân nhóm (nếu bạn muốn vẽ biểu đồ)
# # Chỉ vẽ 2 chiều đầu tiên của dữ liệu (easy và medium) để dễ dàng quan sát
# plt.scatter(data['easy'], data['medium'], c=data['cluster'], cmap='viridis')
# plt.title('K-means Clustering of Sudoku Players')
# plt.xlabel('Time to solve easy Sudoku')
# plt.ylabel('Time to solve medium Sudoku')
# plt.show()

# # 7. Lưu kết quả vào một file CSV
# output_file = 'sudoku_player_classification.csv'
# data.to_csv(output_file, index=False)

# print(f"Kết quả đã được lưu vào file {output_file}")


# import os
# import shutil
# import random

# # Đường dẫn tới thư mục chứa các hình ảnh
# image_folder = 'Repository/digit_images'  # Thay thế đường dẫn đúng
# output_folder = 'shuffer'  # Thư mục lưu ảnh đã xáo trộn

# # Lấy danh sách tất cả các thư mục con chứa hình ảnh
# image_dirs = [d for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))]

# # Tạo thư mục shuffer nếu chưa tồn tại
# os.makedirs(output_folder, exist_ok=True)

# # Mảng để chứa tất cả các hình ảnh từ các thư mục con
# image_files = []

# # Đọc tất cả các hình ảnh từ các thư mục con
# for dir_name in image_dirs:
#     dir_path = os.path.join(image_folder, dir_name)
#     # Lấy tất cả các hình ảnh trong thư mục con
#     files = [f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')]  # Thêm định dạng nếu cần
#     for file in files:
#         image_files.append(os.path.join(dir_path, file))

# # Xáo trộn danh sách hình ảnh
# random.shuffle(image_files)

# # Sao chép các hình ảnh vào thư mục shuffer
# for i, img_path in enumerate(image_files):
#     # Lấy tên tệp gốc
#     file_name = os.path.basename(img_path)
#     # Tạo tên tệp mới để tránh trùng lặp
#     new_path = os.path.join(output_folder, f'{i}_{file_name}')
#     shutil.copy(img_path, new_path)

# print(f"Đã xáo trộn và lưu các hình ảnh vào thư mục {output_folder}.")
import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# --- Bước 1: Định nghĩa hàm đọc hình ảnh ---
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

# --- Bước 2: Tiền xử lý hình ảnh (resize) và trích xuất HOG ---
def extract_hog_features(images):
    hog_features = []
    for img in images:
        # Resize hình ảnh về kích thước 64x64
        img_resized = cv2.resize(img, (64, 64))
        # Trích xuất đặc trưng HOG
        features, _ = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

# --- Bước 3: Phân cụm bằng KMeans ---
def cluster_images(hog_features, n_clusters=9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(hog_features)
    return kmeans.labels_

# --- Bước 4: Lưu hình ảnh với nhãn ---
def save_clustered_images(folder, filenames, labels, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename, label in zip(filenames, labels):
        label_folder = os.path.join(output_folder, f'cluster_{label}')
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        # Copy file vào thư mục tương ứng với nhãn
        img_path = os.path.join(folder, filename)
        output_path = os.path.join(label_folder, filename)
        cv2.imwrite(output_path, cv2.imread(img_path))

# --- Bước 5: Chạy toàn bộ quy trình ---
if __name__ == "__main__":
    input_folder = "shuffer/"       # Thay đổi đường dẫn tới thư mục hình ảnh của bạn
    output_folder = "clusters"  # Thư mục lưu kết quả
    
    # Đọc hình ảnh
    images, filenames = load_images_from_folder(input_folder)
    print(f"Loaded {len(images)} images.")
    
    # Trích xuất HOG
    print("Extracting HOG features...")
    hog_features = extract_hog_features(images)
    
    # Phân cụm bằng KMeans
    print("Clustering images with KMeans...")
    labels = cluster_images(hog_features, n_clusters=9)
    
    # Lưu hình ảnh với nhãn
    print("Saving clustered images...")
    save_clustered_images(input_folder, filenames, labels, output_folder)
    
    print("Clustering complete. Images saved to", output_folder)
