import tkinter as tk
from tkinter import font as tkfont
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import os
import threading
import time
import subprocess
from SudokuApp import run_app
from SudokuLoadingSplash import SplashScreen
from ttkbootstrap.tooltip import ToolTip
import pandas as pd  # Import pandas để làm việc với DataFrame
from sklearn.cluster import KMeans  # Import KMeans từ scikit-learn để thực hiện phân cụm
from sklearn.preprocessing import StandardScaler  # Import StandardScaler để chuẩn hóa dữ liệu
import tkinter as tk  # Thư viện GUI cơ bản
from matplotlib.figure import Figure  # Tạo biểu đồ với matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Kết hợp matplotlib với tkinter
import pandas as pd  # Xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Vẽ biểu đồ
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.dialogs import Dialog
class SudokuHome(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Sudoku Master")
        self.geometry("1024x768")
        self.resizable(False, False)

        self.setup_background()
        self.setup_title()
        self.setup_name_entry()
        self.setup_buttons()
        self.setup_decorations()

    def setup_background(self):
        bg_image = Image.new('RGB', (1024, 768), color='#2C3E50')
        draw = ImageDraw.Draw(bg_image)
        for y in range(768):
            r = int(44 + (y / 768) * (52 - 44))
            g = int(62 + (y / 768) * (73 - 62))
            b = int(80 + (y / 768) * (94 - 80))
            draw.line([(0, y), (1024, y)], fill=(r, g, b))

        # Add some subtle patterns
        pattern = Image.open("Repository/subtle_pattern.jpg").convert('RGBA')
        pattern = pattern.resize((1024, 768), Image.LANCZOS)
        bg_image.paste(pattern, (0, 0), pattern)

        self.bg_photo = ImageTk.PhotoImage(bg_image)

        self.canvas = tk.Canvas(self, width=1024, height=768)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
    def setup_name_entry(self):
        name_frame = ttk.Frame(self)
        name_frame_window = self.canvas.create_window(512, 300, anchor="center", window=name_frame)

        name_label = ttk.Label(name_frame, text="Tên người chơi:", font=("Roboto", 14))
        name_label.pack(side="left", padx=(0, 10))

        self.name_entry = ttk.Entry(name_frame, font=("Roboto", 14), width=20)
        self.name_entry.pack(side="left")

    def setup_title(self):
        title_font = tkfont.Font(family="Roboto", size=64, weight="bold")

        # Create a stylish title with shadow effect
        shadow_color = "#1E1E1E"
        glow_color = "#3498DB"
        main_color = "#ECF0F1"

        text_position = 512, 128
        for i in range(6, 0, -1):
            self.canvas.create_text(text_position[0]+i, text_position[1]+i, text="Sudoku Master", font=title_font, fill=shadow_color)
        
        self.canvas.create_text(text_position[0]-1, text_position[1]-1, text="Sudoku Master", font=title_font, fill=glow_color)
        self.canvas.create_text(text_position[0], text_position[1], text="Sudoku Master", font=title_font, fill=main_color)

    def setup_buttons(self):
        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=("Roboto", 16, "bold"), padding=12)

        # Load button icons
        play_icon = tk.PhotoImage(file="Repository/play_icon.png")
        stats_icon = tk.PhotoImage(file="Repository/stats_icon.png")

        play_button = ttk.Button(self, text="Chơi Game", image=play_icon, compound="left", 
                                  command=self.play_game, style="Custom.TButton", cursor="hand2")
        play_button.image = play_icon
        play_button_window = self.canvas.create_window(512, 384, anchor="center", window=play_button, width=240, height=84)

        stats_button = ttk.Button(self, text="Thống Kê", image=stats_icon, compound="left",
                                  command=self.show_statistics, style="Custom.TButton", cursor="hand2")
        stats_button.image = stats_icon
        stats_button_window = self.canvas.create_window(512, 488, anchor="center", window=stats_button, width=240, height=84)

    def setup_decorations(self):
        # Add Sudoku-themed decorations
        sudoku_grid = Image.open(os.path.join('Repository', 'sudoku.png')).convert('RGBA')
        sudoku_grid = sudoku_grid.resize((192, 192), Image.LANCZOS)
        sudoku_grid = sudoku_grid.filter(ImageFilter.GaussianBlur(radius=1))
        self.sudoku_grid_photo = ImageTk.PhotoImage(sudoku_grid)

        self.canvas.create_image(128, 640, image=self.sudoku_grid_photo, anchor="center")
        self.canvas.create_image(896, 268, image=self.sudoku_grid_photo, anchor="center")

    def play_game(self):
        # Show loading screen before starting the game
        player_name = self.name_entry.get().strip()
        if not player_name:
            Messagebox.show_warning(
               "Vui lòng nhập tên người chơi!",
                "Error", parent=self
            )
            return
        self.withdraw()  # Hide the main window
        loading_screen = SplashScreen()
        loading_screen.after(100, loading_screen.animate)  # Start animation after loading screen is visible
        self.wait_window(loading_screen)  # Wait for the loading screen to close
        self.start_main_game(player_name)


    def start_main_game(self, player_name):
        run_app(player_name)
        # self.destroy()  # Close the home screen
    def show_statistics(self):
        # Đọc file CSV
        df = pd.read_csv('player_data.csv')

        # Chuyển đổi 'Độ khó' thành giá trị số
        difficulty_map = {'Dễ': 1, 'Trung bình': 2, 'Khó': 3, 'Chuyên gia': 4}
        df['Độ khó số'] = df['Difficulty'].map(difficulty_map)

        # Chuẩn bị dữ liệu cho phân cụm
        X = df[['Solving Time', 'Error Count', 'Độ khó số']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Xác định số cụm tối ưu bằng phương pháp elbow
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        # Thực hiện phân cụm K-means với số cụm tối ưu
        optimal_clusters = 3  # Có thể thay đổi dựa trên đường cong elbow
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        df['Cụm'] = kmeans.fit_predict(X_scaled)

        # Tạo cửa sổ mới cho thống kê
        stats_window = ttk.Toplevel(self)
        stats_window.title("Thống kê người chơi")
        stats_window.geometry("1000x800")

        # Tạo Figure và canvas
        fig = plt.figure(figsize=(12, 8))
        canvas = FigureCanvasTkAgg(fig, master=stats_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Vẽ biểu đồ phân cụm
        ax1 = fig.add_subplot(221, projection='3d')
        scatter = ax1.scatter(df['Solving Time'], df['Error Count'], df['Độ khó số'], c=df['Cụm'], cmap='viridis')
        ax1.set_xlabel('Thời gian giải (giây)')
        ax1.set_ylabel('Số lỗi')
        ax1.set_zlabel('Độ khó')
        ax1.set_title('Phân cụm người chơi dựa trên hiệu suất')

        # Thêm tâm cụm
        centers = kmeans.cluster_centers_
        ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=200, linewidths=3)

        # Vẽ đường cong elbow
        ax2 = fig.add_subplot(222)
        ax2.plot(range(1, 11), wcss)
        ax2.set_title('Phương pháp Elbow')
        ax2.set_xlabel('Số cụm')
        ax2.set_ylabel('WCSS')

        # Vẽ biểu đồ thời gian giải trung bình theo độ khó
        ax3 = fig.add_subplot(223)
        df.groupby('Difficulty')['Solving Time'].mean().plot(kind='bar', ax=ax3)
        ax3.set_title('Thời gian giải trung bình theo độ khó')
        ax3.set_ylabel('Thời gian giải trung bình (giây)')
        ax3.set_xlabel('Độ khó')

        # Vẽ biểu đồ phân phối số lỗi
        ax4 = fig.add_subplot(224)
        df['Error Count'].plot(kind='hist', ax=ax4, bins=10)
        ax4.set_title('Phân phối số lỗi')
        ax4.set_xlabel('Số lỗi')
        ax4.set_ylabel('Tần suất')

        # Điều chỉnh bố cục và cập nhật canvas
        plt.tight_layout()
        canvas.draw()

        # Thêm nút đóng
        close_button = ttk.Button(stats_window, text="Đóng", command=stats_window.destroy)
        close_button.pack(pady=10)

        # Hiển thị thông tin cụm
        cluster_info = df.groupby('Cụm').agg({
            'Solving Time': 'mean',
            'Error Count': 'mean',
            'Difficulty': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Hỗn hợp'
        }).round(2)

        cluster_info_text = "Thông tin cụm:\n" + cluster_info.to_string()
        cluster_info_label = ttk.Label(stats_window, text=cluster_info_text, justify=tk.LEFT, font=("Courier", 10))
        cluster_info_label.pack(pady=10)
        
def run_Home():
    app = SudokuHome()
    app.mainloop()

if __name__ == "__main__":
    app = SudokuHome()
    app.mainloop()

