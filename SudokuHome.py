import tkinter as tk
from tkinter import font as tkfont
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import os
import threading
import time
import subprocess
from SudokuApp import run_app
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
class LoadingScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Loading")
        self.geometry("300x150")
        self.resizable(False, False)
        self.configure(bg="black")

        self.label = tk.Label(self, text="Loading...", font=("Helvetica", 18), fg="white", bg="black")
        self.label.pack(expand=True)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=200, mode="indeterminate")
        self.progress.pack(expand=True)
        self.progress.start()

class SudokuHome(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Sudoku Game")
        self.geometry("800x600")
        self.resizable(False, False)

        self.setup_background()
        self.setup_title()
        self.setup_buttons()
        self.setup_decorations()
        

    def setup_background(self):
        # Tải và thay đổi kích thước hình nền
        bg_image = Image.open("Repository/background.jpg")
        bg_image = bg_image.resize((800, 600), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        # Tạo một canvas và đặt hình ảnh nền lên đó
        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

    def setup_title(self):
        #  tạo font custom cho title
        title_font = tkfont.Font(family="Helvetica", size=48, weight="bold")
        
        # thêm title cho canvas
        self.canvas.create_text(400, 100, text="Sudoku Master", font=title_font, fill="white")

    def setup_buttons(self):
        # Load button icons
        play_icon = tk.PhotoImage(file="Repository/play_icon.png")
        stats_icon = tk.PhotoImage(file="Repository/stats_icon.png")

        # tạo nút play game
        play_button = ttk.Button(self, text="Chơi Game", image=play_icon, compound="left", 
                                 command=self.start_main_game, style="large.TButton")
        play_button.image = play_icon
        play_button_window = self.canvas.create_window(400, 300, anchor="center", window=play_button)

        # tạo nút thống kê
        stats_button = ttk.Button(self, text="Thống Kê", image=stats_icon, compound="left", 
                                  command=self.show_statistics, style="large.TButton")
        stats_button.image = stats_icon
        stats_button_window = self.canvas.create_window(400, 380, anchor="center", window=stats_button)

    def setup_decorations(self):
        # Thêm một số yếu tố trang trí (ví dụ: phác thảo lưới Sudoku)
        self.canvas.create_rectangle(50, 50, 750, 550, outline="white", width=2)
        for i in range(1, 3):
            self.canvas.create_line(50 + i * 233, 50, 50 + i * 233, 550, fill="white", width=2)
            self.canvas.create_line(50, 50 + i * 167, 750, 50 + i * 167, fill="white", width=2)

    def load_main_game(self, loading_screen):
        time.sleep(3)
        
        self.after(0, loading_screen.destroy)
        
        self.after(100, self.destroy)
        
        # Start main game khi bấm 
        self.after(200, self.start_main_game)

    def start_main_game(self):
        #run_app()
        subprocess.Popen(["python", "SudokuApp.py"])

 
    def play_game(self):
        # Hàm để mở ứng dụng Sudoku
        #tạo loading screen
        loading_screen = LoadingScreen(self)
        
        
        threading.Thread(target=self.load_main_game, args=(loading_screen,), daemon=True).start()


    def show_statistics(self):
        print("Showing player statistics...")
def run_Home():
    app= SudokuHome()
    app.mainloop()
if __name__ == "__main__":
    app = SudokuHome()
    app.mainloop()

