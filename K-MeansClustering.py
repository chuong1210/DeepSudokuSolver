import csv
import random
import uuid

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def generate_sudoku_data(num_rows=300):
    """
    Generates random Sudoku solving time data for a CSV file.

    Args:
        num_rows: The number of rows to generate.

    Returns:
        A list of lists, where each inner list represents a row of data.
    """

    data = [
        ["Người chơi", "Thời gian giải dễ (phút)", "Thời gian giải trung bình (phút)", "Thời gian giải khó (phút)"]
    ]

    # Define difficulty ranges (adjust as needed)
    easy_range = (1, 10)  # Example: 1 to 10 minutes for easy
    medium_range = (5, 20)  # Example: 5 to 20 minutes for medium
    hard_range = (10, 30)  # Example: 10 to 30 minutes for hard


    for i in range(num_rows):
        player_name = f"Người {i + 1}"  #  A more organized name format
        easy_time = random.randint(*easy_range)
        medium_time = random.randint(max(easy_time, medium_range[0]), medium_range[1])
        hard_time = random.randint(max(medium_time, hard_range[0]), hard_range[1])

        row = [player_name, easy_time, medium_time, hard_time]
        data.append(row)

    return data



def write_data_to_csv(data, filename="sudoku_times.csv"):
    """Writes the generated data to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

# if __name__ == "__main__":
#     generated_data = generate_sudoku_data(300)
#     write_data_to_csv(generated_data)
#     print("Dữ liệu đã được tạo và lưu vào sudoku_times.csv")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Chuẩn bị dữ liệu: Giả sử chúng ta có bảng điểm của 300 người chơi
# Mỗi người chơi sẽ có thời gian giải Sudoku cho 3 mức độ (dễ, trung bình, khó)

# Tạo dữ liệu mẫu giả
np.random.seed(42)  # Để tái tạo lại dữ liệu cho lần sau
num_players = 300

# Giả lập thời gian giải của các người chơi (giới hạn thời gian giải trong phút)
easy_times = np.random.randint(1, 20, size=num_players)  # Thời gian giải cho mức độ dễ (1-20 phút)
medium_times = np.random.randint(5, 30, size=num_players)  # Thời gian giải cho mức độ trung bình (5-30 phút)
hard_times = np.random.randint(10, 40, size=num_players)  # Thời gian giải cho mức độ khó (10-40 phút)

# Tạo bảng dữ liệu
data = pd.DataFrame({
    'easy': easy_times,
    'medium': medium_times,
    'hard': hard_times
})

# 2. Tiến hành chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 3. Áp dụng K-means clustering
# Giả sử chúng ta sẽ phân thành 3 nhóm: "Gà", "Decent", "Pro"
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_data)

# 4. Đánh giá nhóm của người chơi
# Chúng ta có thể gán các nhãn theo các nhóm: "Gà", "Decent", "Pro" dựa trên thời gian giải
def assign_label(row, difficulty):
    if difficulty == 'easy':
        if row['easy'] > 10:
            return 'Gà'
        elif row['easy'] <= 5:
            return 'Pro'
        else:
            return 'Decent'
    elif difficulty == 'medium':
        if row['medium'] > 15:
            return 'Gà'
        elif row['medium'] <= 6:
            return 'Pro'
        else:
            return 'Decent'
    else:  # 'hard'
        if row['hard'] > 20:
            return 'Gà'
        elif row['hard'] <= 9:
            return 'Pro'
        else:
            return 'Decent'

# Áp dụng phân loại dựa trên các mức độ khó
data['easy_label'] = data.apply(lambda row: assign_label(row, 'easy'), axis=1)
data['medium_label'] = data.apply(lambda row: assign_label(row, 'medium'), axis=1)
data['hard_label'] = data.apply(lambda row: assign_label(row, 'hard'), axis=1)

# 5. Kiểm tra kết quả phân nhóm
print(data[['easy', 'medium', 'hard', 'cluster', 'easy_label', 'medium_label', 'hard_label']].head())

# 6. Visualize kết quả phân nhóm (nếu bạn muốn vẽ biểu đồ)
# Chỉ vẽ 2 chiều đầu tiên của dữ liệu (easy và medium) để dễ dàng quan sát
plt.scatter(data['easy'], data['medium'], c=data['cluster'], cmap='viridis')
plt.title('K-means Clustering of Sudoku Players')
plt.xlabel('Time to solve easy Sudoku')
plt.ylabel('Time to solve medium Sudoku')
plt.show()

# 7. Lưu kết quả vào một file CSV
output_file = 'sudoku_player_classification.csv'
data.to_csv(output_file, index=False)

print(f"Kết quả đã được lưu vào file {output_file}")
