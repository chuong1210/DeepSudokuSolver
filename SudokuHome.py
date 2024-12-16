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

class SudokuHome(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Sudoku Master")
        self.geometry("1024x768")
        self.resizable(False, False)

        self.setup_background()
        self.setup_title()
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
        self.withdraw()  # Hide the main window
        loading_screen = SplashScreen()
        loading_screen.after(100, loading_screen.animate)  # Start animation after loading screen is visible
        self.wait_window(loading_screen)  # Wait for the loading screen to close
        self.start_main_game()

    def start_main_game(self):
        run_app()
        self.destroy()  # Close the home screen

    def show_statistics(self):
        print("Showing player statistics...")

def run_Home():
    app = SudokuHome()
    app.mainloop()

if __name__ == "__main__":
    app = SudokuHome()
    app.mainloop()

