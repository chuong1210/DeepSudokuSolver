import csv
import random
from datetime import datetime, timedelta

def generate_synthetic_data(filename, num_entries=1000):
    difficulties = ["Dễ", "Trung bình", "Khó", "Chuyên gia"]
    vietnamese_names = [
        "Nguyễn Văn A", "Trần Thị B", "Lê Văn C", "Phạm Thị D", "Hoàng Văn E",
        "Đặng Thị F", "Bùi Văn G", "Đỗ Thị H", "Hồ Văn I", "Ngô Thị K",
        "Dương Văn L", "Lý Thị M", "Vũ Văn N", "Đặng Thị O", "Trịnh Văn P"
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Player Name", "Difficulty", "Solving Time", "Error Count", "Date"])

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Data for the last year

        for _ in range(num_entries):
            name = random.choice(vietnamese_names)
            difficulty = random.choice(difficulties)
            
            # Adjust solving time and error count based on difficulty
            if difficulty == "Dễ":
                solving_time = random.uniform(60, 300)  # 1-5 minutes
                error_count = random.randint(0, 5)
            elif difficulty == "Trung bình":
                solving_time = random.uniform(180, 600)  # 3-10 minutes
                error_count = random.randint(0, 10)
            elif difficulty == "Khó":
                solving_time = random.uniform(300, 1200)  # 5-20 minutes
                error_count = random.randint(0, 15)
            else:  # Chuyên gia
                solving_time = random.uniform(600, 1800)  # 10-30 minutes
                error_count = random.randint(0, 20)

            date = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

            writer.writerow([
                name,
                difficulty,
                f"{solving_time:.2f}",
                error_count,
                date.strftime("%Y-%m-%d %H:%M:%S")
            ])

    print(f"Generated {num_entries} entries of synthetic data in {filename}")

# Usage
generate_synthetic_data("player_data.csv")

