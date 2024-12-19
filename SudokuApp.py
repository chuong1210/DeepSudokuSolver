import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import json
import numpy as np
import cv2 
import csv
from datetime import datetime
from PIL import Image, ImageTk
import random
import time
from tkinter import filedialog

from sudokuSolverDiversity import solveAStar, solveBFS, solveDFS,solveIDS,is_valid_sudoku
from ImageProcess import extract_sudoku_grid,displayImageSolution,resize_image,is_sudoku_present,display_sudoku_on_frame

import math
import tkinter.simpledialog as simpledialog

import threading  # Để chạy camera trong một luồng riêng
from DiTruyen import solveGA   # Import the genetic solver
from ttkbootstrap.tooltip import ToolTip
from ttkbootstrap.dialogs import Messagebox

import requests
from io import BytesIO

class SudokuApp(ttk.Toplevel):
    def __init__(self, file,player_name):
        super().__init__()
        self.style.theme_use('darkly')  # Set the theme using ttkbootstrap

        self.title("Sudoku Solver")
        self.geometry("1200x800") # Updated window size

        self.grid_size = 9
        self.original_grid = np.zeros((9, 9), dtype=int)
        self.editable_grid = np.zeros((9, 9), dtype=int)
        self.solution_grid = None
        self.history = []
        self.solution_displayed = False # Added attribute
        self.solution_check=False
        self.hint_mode = False  # Thêm biến để kiểm tra chế độ gợi ý
        self.hinted_cells = []  # Thêm danh sách các ô đã được gợi ý


        self.camera_running = False
        self.elapsed_time = 0

        self.camera_index = 0
        self.max_retries = 3
        self.last_extract_time = time.time()
        self.extract_interval = 5  # Đặt kho

       # ... (previous initialization code remains unchanged)
        self.player_name = player_name
        self.error_count = 0
        self.csv_file = "player_data.csv"
        self.load_db(file)
        self.setup_ui()
    def load_db(self, file):
       
        
        if self.grid_size == 9:
            with open(file) as f:
                data = json.load(f)
            self.easy = data['Easy']
            self.medium = data['Medium']
            self.hard = data['Hard']
            self.expert = data['Expert']
        elif self.grid_size == 16:
            with open("Crossover16.json") as f:
                data = json.load(f)
            self.easy = data['Easy16']
            self.medium = data['Medium16']
            self.hard = data['Hard16']
            self.expert = data['Expert16']


    def setup_ui(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=LEFT, fill=BOTH, expand=YES)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=RIGHT, fill=BOTH, expand=YES, padx=(10, 0))

        self.create_sudoku_grid()
        self.create_control_panel()

    def create_sudoku_grid(self):
        self.canvas = tk.Canvas(self.left_frame, width=700, height=700, bg='white') # Updated canvas size

        self.canvas.pack(fill=BOTH, expand=YES)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        w, h = 700, 700 # Updated canvas size

        padding_top = 40  # Add 50 pixels of padding at the top
        self.cell_size = w / self.grid_size

        for i in range(self.grid_size + 1):
            line_width = 3 if i % int(math.sqrt(self.grid_size)) == 0 else 1
            self.canvas.create_line(i * self.cell_size, padding_top, i * self.cell_size, h + padding_top, width=line_width)
            self.canvas.create_line(0, i * self.cell_size + padding_top, w, i * self.cell_size + padding_top, width=line_width)

        self.cells = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x, y = col * self.cell_size, row * self.cell_size + padding_top
                cell = self.canvas.create_rectangle(x, y, x + self.cell_size, y + self.cell_size, fill="white")
                text = self.canvas.create_text(x + self.cell_size/2, y + self.cell_size/2, text="", font=("Arial", 18))
                self.cells[row][col] = (cell, text)
                self.canvas.tag_bind(cell, "<Button-1>", lambda event, r=row, c=col: self.cell_clicked(r, c))

    # def cell_clicked(self, row, col):
    #     if self.original_grid[row][col] == 0:
    #         value = simpledialog.askinteger("Nhập", f"Nhập giá trị cho ô ({row+1}, {col+1}):", 
    #                                         minvalue=1, maxvalue=self.grid_size)
    #         if value is not None:
    #             self.history.append((row, col, self.editable_grid[row][col]))
    #             self.editable_grid[row][col] = value
    #             self.update_cell(row, col, value)

    # def update_cell(self, row, col, value):
    #     cell, text = self.cells[row][col]
    #     self.canvas.itemconfig(text, text=str(value) if value != 0 else "")
    #     fill_color = "lightblue" if self.original_grid[row][col] != 0 else "white"
    #     self.canvas.itemconfig(cell, fill=fill_color)
    def cell_clicked(self, row, col):
        if self.hint_mode:  # Nếu đang ở chế độ gợi ý
            self.show_hint(row, col)
        elif self.original_grid[row][col] == 0:
            value = simpledialog.askinteger("Nhập", f"Nhập giá trị cho ô ({row+1}, {col+1}):",
                                            minvalue=1, maxvalue=self.grid_size)
            if value is not None:
                self.history.append((row, col, self.editable_grid[row][col]))
                self.editable_grid[row][col] = value
                self.update_cell(row, col, value)
    def update_cell(self, row, col, value):
        cell, text = self.cells[row][col]
        self.canvas.itemconfig(text, text=str(value) if value != 0 else "")
        if (row, col) in self.hinted_cells:
            fill_color = "yellow"  # Ô đã được gợi ý sẽ có màu vàng
        else:
            fill_color = "lightblue" if self.original_grid[row][col] != 0 else "white"
        self.canvas.itemconfig(cell, fill=fill_color)
    def toggle_hint_mode(self): # Hàm bật/tắt chế độ gợi ý
        if np.all(self.original_grid == 0):
            Messagebox.show_error(
                "Không thể bật chế độ gợi ý khi bàn cờ trống. Vui lòng tạo một trò chơi mới trước.",
                "Lỗi",
                parent=self
            )
            return
        self.hint_mode = not self.hint_mode
        if self.hint_mode:
            self.canvas.config(cursor="question_arrow") # Thay đổi cursor
            self.info_label.config(text="Chế độ gợi ý: BẬT. Nhấp vào ô để xem gợi ý.")

        else:
            self.canvas.config(cursor="") # Trả cursor về mặc định
            self.info_label.config(text="")
    def show_hint(self, row, col): # Hàm hiển thị gợi ý
        if self.solution_grid is None:
            self.solution_grid = solveIDS(np.copy(self.original_grid))


        if self.solution_grid is not None and self.original_grid[row][col] == 0:
            self.editable_grid[row][col] = self.solution_grid[row][col]
            self.hinted_cells.append((row, col))  # Lưu ô đã gợi ý
            self.update_cell(row, col, self.solution_grid[row][col])

    def create_control_panel(self):
        control_frame = ttk.Frame(self.right_frame)
        control_frame.pack(fill=X, expand=YES, pady=10)

        ttk.Label(control_frame, text="Độ khó:", font=("Roboto",14)).pack(side=LEFT) # Updated font size

        self.difficulty_var = tk.StringVar(value="Dễ")
        difficulty_combo = ttk.Combobox(control_frame, textvariable=self.difficulty_var, 
                                        values=["Dễ", "Trung bình", "Khó", "Chuyên gia"])
        difficulty_combo.pack(side=LEFT, padx=(0, 10))

        ttk.Button(control_frame, text="Trò chơi mới", command=self.new_game,cursor="hand2", width=30).pack(side=LEFT)

        algorithm_frame = ttk.Frame(self.right_frame)
        algorithm_frame.pack(fill=X, expand=YES, pady=10)


        ttk.Label(algorithm_frame, text="Thuật toán:", font=("Roboto",14)).pack(side=LEFT) # Updated font size

        self.algorithm_var = tk.StringVar(value="A*")
        algorithm_combo = ttk.Combobox(algorithm_frame, textvariable=self.algorithm_var, 
                                       values=["A*", "DFS", "BFS", "GA","IDS"])
        algorithm_combo.pack(side=LEFT, padx=(0, 10))



    # ... (rest of the class implementation remains unchanged)
        self.solve_button = ttk.Button(algorithm_frame, text="Giải", command=self.solve,cursor="hand2", width=30) # Updated line
        self.solve_button.pack(side=LEFT)


        grid_size_frame = ttk.Frame(self.right_frame)
        grid_size_frame.pack(fill=X, expand=YES, pady=20)  # Increased vertical padding

        ttk.Label(grid_size_frame, text="Kích thước lưới:", font=("Arial", 14)).pack(side=LEFT, padx=(0, 20))  # Increased right padding
        self.grid_size_var = tk.StringVar(value="9x9")
        grid_size_9x9 = ttk.Radiobutton(grid_size_frame, text="9x9", variable=self.grid_size_var, value="9x9", command=self.change_grid_size, style="TRadiobutton", cursor="hand2")
        grid_size_16x16 = ttk.Radiobutton(grid_size_frame, text="16x16", variable=self.grid_size_var, value="16x16", command=self.change_grid_size, style="TRadiobutton", cursor="hand2")
        grid_size_9x9.pack(side=LEFT, padx=(0, 20))  # Increased spacing between radio buttons
        grid_size_16x16.pack(side=LEFT)

        # Configure the radio button style
        style = ttk.Style()
        style.configure("TRadiobutton", font=("Arial", 12), padding=10)  # Increased font size and padding

        timer_frame = ttk.Frame(self.right_frame)
        timer_frame.pack(fill=X, expand=YES, pady=10)
        self.timer_label = ttk.Label(timer_frame, text="Thời gian: 00:00",font=( 16))
        self.timer_label.pack(side=LEFT)
        self.start_time = time.time()
        self.timer_running = False
        self.update_timer()



        # Load icons
        undo_icon_image = Image.open("sudoku_images/icon.png").resize((30, 30), Image.LANCZOS) # Updated icon size
        self.undo_icon = ImageTk.PhotoImage(undo_icon_image)
        refresh_icon_image = Image.open("Repository/refresh_icon.png").resize((30, 30), Image.LANCZOS) # Updated icon size and path
        self.refresh_icon = ImageTk.PhotoImage(refresh_icon_image)
        open_icon_image = Image.open("Repository/open.png").resize((30, 30), Image.LANCZOS) # Updated icon size and path
        self.open_icon = ImageTk.PhotoImage(open_icon_image)
        close_icon_image = Image.open("Repository/close.png").resize((30, 30), Image.LANCZOS) # Updated icon size and path
        self.close_icon = ImageTk.PhotoImage(close_icon_image)
        hint_icon_image = Image.open("Repository/hint_icon.png").resize((30, 30), Image.LANCZOS) # Updated icon size and path
        self.hint_icon = ImageTk.PhotoImage(hint_icon_image)
        upload_icon_image = Image.open("Repository/camera.png").resize((30, 30), Image.LANCZOS) # Updated icon size and path
        self.upload_icon = ImageTk.PhotoImage(upload_icon_image)
        check_icon_image = Image.open("Repository/check_icon.png").resize((30, 30), Image.LANCZOS) # Updated icon size and path
        self.check_icon = ImageTk.PhotoImage(check_icon_image)



  
        # Nút Undo với icon
        undo_frame = ttk.Frame(self.right_frame)
        undo_frame.pack(fill=X, expand=YES, pady=10)
        undo_button = ttk.Button(undo_frame, text="Hoàn tác", image=self.undo_icon, compound=LEFT, command=self.undo_move,width=20,cursor="hand2")
        undo_button.pack(side=LEFT,padx=5)

  

        # Nút Mở Camera
        self.open_camera_button = ttk.Button(undo_frame, text="Open Camera", image=self.open_icon, command=self.show_camera_menu, width=20, cursor="hand2")
        self.open_camera_button.pack(side=LEFT, padx=5)
        ToolTip(self.open_camera_button, text="Open Camera")

        # Nút Đóng Camera
        close_camera_button = ttk.Button(undo_frame, text="Đóng Camera", image=self.close_icon,command=self.close_camera, width=20,cursor="hand2")
        close_camera_button.pack(side=LEFT, padx=5)
        ToolTip(close_camera_button, text="Tắt Camera") # Thêm tooltip




        hint_button = ttk.Button(undo_frame, text="Gợi ý", image=self.hint_icon,  command=self.toggle_hint_mode, width=20, cursor="hand2")
        hint_button.pack(side=LEFT, padx=5)
        ToolTip(hint_button, text="Bật/tắt chế độ gợi ý") # Thêm tooltip
      

        solution_frame = ttk.Frame(self.right_frame)
        solution_frame.pack(fill=X, expand=YES, pady=10)

        upload_image_button=ttk.Button(self.right_frame, text="Tải lên hình ảnh", image=self.upload_icon,command=self.upload_image,compound=RIGHT,cursor="hand2")
        upload_image_button.pack(fill=X, expand=YES, pady=10)





        # Create buttons with icons
        solution_button = ttk.Button(solution_frame, text="Kiểm tra đáp án", image=self.check_icon, compound=LEFT, command=self.check_solution, width=30,cursor="hand2")
        solution_button.pack(side=LEFT, expand=YES, padx=(0, 5))

        refresh_button = ttk.Button(solution_frame, text="Làm mới", image=self.refresh_icon, compound=LEFT, command=self.refresh_grid, width=30,cursor="hand2")
        refresh_button.pack(side=LEFT, expand=YES, padx=(5, 0))


        self.info_label = ttk.Label(self.right_frame, text="",font=(14))
        self.info_label.pack(fill=X, expand=YES, pady=10)
        # Thêm thah progress bar
        self.progress_bar = ttk.Progressbar(self.right_frame, orient=HORIZONTAL, length=300, mode='determinate', style="TProgressbar")
        self.progress_bar.pack(fill=X, expand=YES, pady=10)
        
        # Configure the progress bar style
        style = ttk.Style()
        style.configure("TProgressbar", thickness=20, troughcolor='#E0E0E0', background='#4CAF50')


    def new_game(self):
        level = self.difficulty_var.get()
        if self.grid_size == 9:
            self.load_db("Crossover.json")

            if level == "Dễ":
                puzzle = random.choice(self.easy)
            elif level == "Trung bình":
                puzzle = random.choice(self.medium)
            elif level == "Khó":
                puzzle = random.choice(self.hard)
            elif level == "Chuyên gia":
                puzzle = random.choice(self.expert)
            else:
                puzzle = "0" * 81  #  cho 1 grid trống nếu invalid
            puzzle = list(map(int, puzzle))

            self.original_grid = np.array(puzzle).reshape((9, 9))
            self.editable_grid = np.copy(self.original_grid)
        elif self.grid_size == 16:
            self.load_db("Crossover.json")


            if level == "Dễ":
                puzzle = random.choice(self.easy)
            elif level == "Trung bình":
                puzzle = random.choice(self.medium)
            elif level == "Khó":
                puzzle = random.choice(self.hard)
            elif level == "Chuyên gia":
                puzzle = random.choice(self.expert)
            else:
                puzzle = "0" * 256  # Empty grid if level is invalid
            puzzle = [x.replace('G', '10') for x in puzzle]
            puzzle = list(map(lambda x: int(x, 16), puzzle))  # Convert from base 16
            print(len(puzzle))
            # Ensure that the puzzle has exactly 256 numbers, and reshape it to (16, 16)
            if len(puzzle) == 256:
                self.original_grid = np.array(puzzle).reshape((16, 16))
                self.editable_grid = np.copy(self.original_grid)
            else:
                print("Error: Puzzle for 16x16 grid is not valid.")
        # Đảm bảo câu đố được chuyển đổi thành danh sách các số nguyên  
        self.new_game_action()



    def new_game_action(self):
        self.solution_grid = None
        self.history = []
        self.hint_mode = False  # Thêm biến để kiểm tra chế độ gợi ý
        self.hinted_cells = []  # Thêm danh sách các ô đã được gợi ý
        self.update_grid_display()
        
        # Đặt lại và bắt đầu hẹn giờ từ đầu
        self.timer_running = False
        self.start_time = time.time()
        self.timer_label.config(text="Thời gian: 00:00")
        self.info_label.config(text="")

        self.timer_running = True
        
        # Đặt lại và bắt đầu progress bar
        self.progress_bar.stop()
        self.progress_bar['value'] = 0
        self.progress_bar.start()
        self.progress_bar['maximum'] = 100
        self.elapsed_time = 0
        
        self.update_timer()
        self.solution_displayed = False # Added line
        self.update_solve_button_text() # Added line

    def update_grid_display(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.update_cell(row, col, self.editable_grid[row][col])

    def solve(self):
        if np.all(self.original_grid == 0):
            Messagebox.show_warning("Vui lòng tạo một trò chơi mới trước khi giải.", "Lỗi", parent=self)
            return
        if not self.solution_displayed:
            self.timer_running = False
            self.progress_bar.stop()
            algorithm = self.algorithm_var.get()
            if algorithm == "GA":
                if self.grid_size == 9:
                    # s = gss.Sudoku(9)
                    # s.load(self.original_grid)
                    # _, solution = s.solve()
                    # self.solution_grid = solution.values if solution else None
                    self.solution_grid = solveGA(np.copy(self.original_grid))

                else:
                    # s = gss.Sudoku(16)
                    # s.load(self.original_grid)
                    # _, solution = s.solve()
                    # self.solution_grid = solution.values if solution else None
                    self.solution_grid  =solveGA(np.copy(self.original_grid), population_size=2000, repetitions=2000, pm=0.2, pc=0.9)


            elif algorithm == "DFS":
                    
                if self.grid_size == 9:
                    self.solution_grid = solveDFS(np.copy(self.original_grid))
                else:
                    self.solution_grid = solveDFS(np.copy(self.original_grid),16)

            elif algorithm == "BFS":
                if self.grid_size == 9:

                    self.solution_grid = solveBFS(np.copy(self.original_grid))
                else:
                    self.solution_grid = solveBFS(np.copy(self.original_grid),16)
            elif algorithm == "A*":
                if self.grid_size == 9:

                    self.solution_grid = solveAStar(np.copy(self.original_grid))
                else:
                    self.solution_grid = solveAStar(np.copy(self.original_grid),16)
                
        
            else:
                if self.grid_size == 9:
                    self.solution_grid = solveIDS(np.copy(self.original_grid))
                else:
                    self.solution_grid = solveIDS(np.copy(self.original_grid),16)
            

            if self.solution_grid is not None:
                self.info_label.config(text="Đang hiển thị đáp án...")
                self.display_solution_gradually()
                self.solution_displayed = True

            else:
                self.info_label.config(text="Không tìm thấy đáp án.")
        else:
            self.hide_solution()
            self.solution_displayed = False
        
        self.update_solve_button_text()

    def display_solution_gradually(self):
        self.solution_display_index = 0
        self.display_next_solution_cell()

    def display_next_solution_cell(self):
        if self.solution_display_index < self.grid_size * self.grid_size:
            row = self.solution_display_index // self.grid_size
            col = self.solution_display_index % self.grid_size
            if self.original_grid[row][col] == 0:
                self.editable_grid[row][col] = self.solution_grid[row][col]
                self.update_cell(row, col, self.solution_grid[row][col])
            self.solution_display_index += 1
            self.after(10, self.display_next_solution_cell) #50 nếu muốn chậm hơn
        else:
            self.info_label.config(text="Câu đố đã được giải thành công!")


    def change_grid_size(self):
        size = self.grid_size_var.get()
        self.grid_size = 9 if size == "9x9" else 16
        self.original_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.editable_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.solution_grid = None
        self.history = []
        self.draw_grid()

    def upload_image(self):
        # Chọn tệp hình ảnh
        file_path = filedialog.askopenfilename(filetypes=[("Tệp hình ảnh", "*.jpg;*.png;*.gif")])
        if not file_path:
            Messagebox.show_info("Vui lòng chọn một tệp hình ảnh.", "Lưu ý", parent=self)
            return
        image = cv2.imread(file_path)

        if file_path:
            # Giải Sudoku từ ảnh
            sudoku_grid, largest_rect_coord, transf, (maxWidth, maxHeight) = extract_sudoku_grid(image, "models/model_sudoku.keras")

            self.original_grid = np.array(sudoku_grid)
            self.editable_grid = np.copy(self.original_grid)
            self.solution_grid = None
            self.history = []
            self.update_grid_display()


            # Xác nhận câu đố Sudoku
            if Messagebox.yesno("Đây có phải là câu đố Sudoku chính xác không?", "Xác nhận", parent=self) and is_valid_sudoku(self.original_grid):
                self.new_game_action()
             

                self.solve()
                   
                print(1)
                print(2)

                print(self.original_grid)
                print(self.solution_grid)

                # Hiển thị ảnh với lời giải
                if(self.solution_grid is not None):

                    result_image = displayImageSolution(image, self.solution_grid, self.original_grid,largest_rect_coord)

                    result_image = resize_image(result_image)  # Resize the image if it's too large

                    cv2.imshow("Sudoku Solved", result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                Messagebox.show_info("Vui lòng tải lên một hình ảnh khác.", "Thử lại", parent=self)

    def show_camera_menu(self):
        camera_menu = tk.Menu(self, tearoff=0,cursor="hand2")
        camera_menu.add_command(label="Open Webcam", command=self.open_webcam)
        camera_menu.add_command(label="Open Phone Camera", command=self.open_phone_camera)
    
        # Get the position of the "Open Camera" button
        button_x = self.open_camera_button.winfo_rootx()
        button_y = self.open_camera_button.winfo_rooty() + self.open_camera_button.winfo_height()
    
        # Display the menu at the button's position
        camera_menu.tk_popup(button_x, button_y)

    def start_camera_feed(self, cap):
        if not cap.isOpened():
            Messagebox.show_error(f"Không thể kết nối với camera:","Lỗi" , parent=self)
            return

        self.camera_window = tk.Toplevel(self)
        self.camera_window.title("Real-Time Sudoku Camera")
        self.camera_window.geometry("800x600")

        self.camera_frame = ttk.Frame(self.camera_window)
        self.camera_frame.pack(fill=tk.BOTH, expand=tk.YES)

        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=tk.YES)
        self.camera_running = True
        #threading.Thread(target=self.update_camera_feed_myself, args=(cap,), daemon=True).start()
        self.camera_window.bind("<Key>", self.on_key_press)

        self.update_camera_feed_myself(cap)


  


    def open_webcam(self):
        # Existing webcam opening logic
        cap = cv2.VideoCapture(0)
        self.start_camera_feed(cap)

    def open_phone_camera(self):
        # Phone camera opening logic
        ip_camera_url = "http://192.168.1.4:8080/video"  # Update with your phone's IP camera URL
        cap = cv2.VideoCapture(ip_camera_url)
        self.start_camera_feed(cap)

    def update_camera_feed_myself(self, cap):
        sudoku_detected = False  # Cờ để kiểm tra trạng thái Sudoku
        largest_rect_coord = None  # Lưu trữ tọa độ của vùng Sudoku
        self.previous_grid = None  # Lưu trữ lưới Sudoku trước đó để so sánh
        self.is_present=False
        sudoku_solve_grid=None

  
        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                Messagebox.show_error("Không thể đọc khung hình từ camera.", "Lỗi", parent=self)
                self.close_camera()
                break

            # Thoát vòng lặp khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Kiểm tra xem có Sudoku trong khung hình không
            if is_sudoku_present(frame) == True :
                sudoku_detected = True
                if sudoku_detected:
                    try:
                        
                        largest_rect_coord, frame_with_sudoku, sudoku_grid, inv_transf, maxWidth, maxHeight = display_sudoku_on_frame(frame, "models/model_sudoku.keras",self.is_present)
                        sudoku_detected = True
                        frame = frame_with_sudoku
                        print(inv_transf)

                        # Tô màu và vẽ viền vùng Sudoku
                        if largest_rect_coord is not None and len(largest_rect_coord) == 4:
                            cv2.drawContours(frame, [largest_rect_coord], -1, (0, 255, 0), -1)  # Tô màu xanh lá
                            cv2.drawContours(frame, [largest_rect_coord], -1, (0, 0, 255), 3)  # Viền đỏ

                        # Kiểm tra và giải Sudoku chỉ khi lưới thay đổi
                        if sudoku_grid is not None and is_valid_sudoku(sudoku_grid) == True:
                            if self.previous_grid is None or not np.array_equal(sudoku_grid, self.previous_grid):
                                self.original_grid = sudoku_grid
                                self.editable_grid = np.copy(self.original_grid)
                                self.previous_grid = np.copy(self.original_grid)  # Lưu lại grid hiện tại
                                self.update_grid_display()
                                self.is_present=True

                                # Giải Sudoku và lưu vào solution_grid
                                self.solution_grid = solveDFS(np.copy(self.original_grid))
                                last_solution_time = time.time()  # Cập nhật thời gian hiển thị kết quả

                                print(self.solution_grid)
                            else:
                                print("Lưới chưa thay đổi, không giải lại Sudoku.")

                        # Vẽ kết quả Sudoku lên khung hình
                        if self.solution_grid is not None:
                            grid_cell_height = maxHeight // 9
                            grid_cell_width = maxWidth // 9

                            for i in range(9):
                                for j in range(9):
                                    # Chỉ vẽ số lên các ô chưa có số trong lưới đề bài
                                    if self.original_grid[i][j] != -1:  # Ô chưa có giá trị trong lưới gốc
                                        num = self.solution_grid[i][j]
                                        if num != -1:
                                            # Tính tọa độ trung tâm của mỗi ô trong khung hình đã chuyển
                                            x_warp = int(j * grid_cell_width + grid_cell_width / 2)
                                            y_warp = int(i * grid_cell_height + grid_cell_height / 2)

                                            # Chuyển đổi tọa độ từ không gian warp về frame gốc
                                            point_warp = np.array([[[x_warp, y_warp]]], dtype=np.float32)  # Đảm bảo là (1, 1, 2)
                                            
                                            # Kiểm tra biến inv_transf có phải là ma trận đồng nhất không
                                            if inv_transf is not None and inv_transf.shape[0] == 3 and inv_transf.shape[1] == 3:
                                                point_frame = cv2.perspectiveTransform(point_warp, inv_transf)[0][0]

                                                # Vẽ số vào ô tương ứng trên frame
                                                cv2.putText(frame, str(num), (int(point_frame[0]), int(point_frame[1])),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                            else:
                                                print("inv_transf không hợp lệ.")

                    except Exception as e:
                        Messagebox.show_error(f"Đã xảy ra lỗi: {e}", "Lỗi", parent=self)
            else:
                # Nếu không phát hiện Sudoku, không làm gì thêm
                sudoku_detected = False

            # Chuyển đổi hình ảnh để hiển thị trong Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (800, 600))
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Cập nhật hình ảnh trong Label
            self.camera_label.imgtk = img_tk
            self.camera_label.configure(image=img_tk)

            # Cập nhật giao diện Tkinter
            self.camera_window.update_idletasks()
            self.camera_window.update()

            # cv2.waitKey(100)  # Thời gian trì hoãn 100ms (hoặc có thể thay đổi theo nhu cầu)

        # Giải phóng tài nguyên khi kết thúc
        cap.release()




    def close_camera(self):
        self.camera_running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'camera_window'):
            self.camera_window.destroy()

 
    def check_solution(self):
        if np.all(self.original_grid == 0):
            Messagebox.show_warning(
               "Tạo 1 game mới trước khi check",
                "Error", parent=self
            )
        if  self.solution_check == False:
            self.solution_check=True

            if self.solution_grid is None:
                # if not self.solution_displayed:
                #     self.timer_running = False
                #     self.progress_bar.stop()
                algorithm = self.algorithm_var.get()
                if algorithm == "GA":
                    if self.grid_size == 9:
                
                        self.solution_grid = solveGA(np.copy(self.original_grid))

                    else:
            
                        self.solution_grid  =solveGA(np.copy(self.original_grid), population_size=2000, repetitions=2000, pm=0.2, pc=0.9)


                elif algorithm == "DFS":
                        
                    if self.grid_size == 9:
                        self.solution_grid = solveDFS(np.copy(self.original_grid))
                    else:
                        self.solution_grid = solveDFS(np.copy(self.original_grid),16)

                elif algorithm == "BFS":
                    if self.grid_size == 9:

                        self.solution_grid = solveBFS(np.copy(self.original_grid))
                    else:
                        self.solution_grid = solveBFS(np.copy(self.original_grid),16)
                elif algorithm == "A*":
                    if self.grid_size == 9:

                        self.solution_grid = solveAStar(np.copy(self.original_grid))
                    else:
                        self.solution_grid = solveAStar(np.copy(self.original_grid),16)
          
                else:
                    if self.grid_size == 9:
                        self.solution_grid = solveIDS(np.copy(self.original_grid))
                    else:
                        self.solution_grid = solveIDS(np.copy(self.original_grid),16)
            if self.solution_grid is not None:
                is_correct = True

                for row in range(self.grid_size):
                    for col in range(self.grid_size):
                        cell, _ = self.cells[row][col]
                        if self.original_grid[row][col] == 0:
                            if self.editable_grid[row][col] == self.solution_grid[row][col]:
                                self.canvas.itemconfig(cell, fill="lightgreen")
                            elif self.editable_grid[row][col] != 0:
                                self.canvas.itemconfig(cell, fill="lightcoral")
                                is_correct = False
                                self.error_count += 1


                            else:
                                self.canvas.itemconfig(cell, fill="white")
                                is_correct = False
                
                if is_correct:
                    solving_time = time.time() - self.start_time
                    self.save_player_data(solving_time)
                    Messagebox.show_info(f"Người chơi {self.player_name} đã giải đúng Sudoku trong {solving_time:.2f} giây!","Chúc mừng! " , parent=self)
                else:
                    Messagebox.show_warning("Có một số ô chưa đúng. Hãy kiểm tra lại!", "Chưa chính xác", parent=self)
            else:
                Messagebox.show_warning("Không tìm thấy đáp án hợp lệ.", "Error", parent=self)



                            


            # if self.solution_grid is not None:
            #         self.info_label.config(text="Đang hiển thị đáp án...")
            #         # self.display_solution_gradually()
            #         # self.solution_displayed = True

            # else:
            #         self.info_label.config(text="Không tìm thấy đáp án.")
        elif(self.solution_check  == True):
                print(self.solution_check)
                self.refresh_checkSol()
                self.solution_check = False
        
        # self.update_solve_button_text()
    

    
    def undo_move(self):
        if self.history:
            row, col, prev_value = self.history.pop()
            self.editable_grid[row][col] = prev_value
            self.update_cell(row, col, prev_value)

    def update_timer(self):
        if self.timer_running:


            elapsed_time = time.time() - self.start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            self.timer_label.config(text=f"Thời gian: {minutes:02d}:{seconds:02d}")
            self.elapsed_time += 1
            self.progress_bar['value'] = (self.elapsed_time % 3600) / 36 * 100  # Scale to 0-100 range
            self.master.after(1000, self.update_timer) #update every second    
    def refresh_checkSol(self):
        self.solution_displayed=  False
        self.solution_check=  False
        self.update_solve_button_text()
        self.hint_mode = False  # Thêm biến để kiểm tra chế độ gợi ý
        self.hinted_cells = []  # Thêm danh sách các ô đã được gợi ý
        self.update_grid_display()
    def refresh_grid(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.original_grid[row][col] == 0:
                    self.editable_grid[row][col] = 0
                    self.update_cell(row, col, 0)
        self.history = []
        self.solution_displayed=  False
        self.update_solve_button_text()
        self.hint_mode = False  # Thêm biến để kiểm tra chế độ gợi ý
        self.hinted_cells = []  # Thêm danh sách các ô đã được gợi ý
        self.update_grid_display()
    def hide_solution(self): # Added method
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.original_grid[row][col] == 0:
                    self.editable_grid[row][col] = 0
                    self.update_cell(row, col, 0)
        self.info_label.config(text="Đáp án đã được ẩn.")

    def update_solve_button_text(self): # Added method
        if self.solution_displayed:
            self.solve_button.config(text="Ẩn đáp án")
        else:
            self.solve_button.config(text="Giải")

    def save_player_data(self, solving_time):
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.player_name,
                self.difficulty_var.get(),
                solving_time,
                self.error_count,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    def on_key_press(self, event):
        if event.char == 'q':
            self.previous_grid = None  # Reset previous_grid về None
            self.camera_running = True  # Dừng vòng lặp camera
            self.is_present=False

            print("Reset previous_grid và dừng vòng lặp.")


def run_app(player_name):
    print(player_name)
    app = SudokuApp("Crossover.json", player_name)
    app.mainloop()
if __name__ == "__main__":
    # This block won't be used when called from SudokuHome
    player_name = "Test Player"
    app = SudokuApp("Crossover.json", player_name)
    app.mainloop()