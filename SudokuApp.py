import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import json
import numpy as np
import cv2
from PIL import Image, ImageTk
import random
import time
from tkinter import filedialog, messagebox
import GA as gss

from sudokuSolverAstart import solveAStar, solveBFS, solveDFS,solveIDS
from ImageProcess import extrapolate_sudoku
import math
import tkinter.simpledialog as simpledialog
import threading  # Để chạy camera trong một luồng riêng

class SudokuApp(ttk.Window):
    def __init__(self, file):
        super().__init__(themename="darkly")
        self.title("Sudoku Solver")
        self.geometry("1000x600")

        self.grid_size = 9
        self.original_grid = np.zeros((9, 9), dtype=int)
        self.editable_grid = np.zeros((9, 9), dtype=int)
        self.solution_grid = None
        self.history = []
        self.solution_displayed = False # Added attribute

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
        self.canvas = tk.Canvas(self.left_frame, width=500, height=500, bg='white')
        self.canvas.pack(fill=BOTH, expand=YES)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        w, h = 500, 500
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

    def cell_clicked(self, row, col):
        if self.original_grid[row][col] == 0:
            value = simpledialog.askinteger("Nhập", f"Nhập giá trị cho ô ({row+1}, {col+1}):", 
                                            minvalue=1, maxvalue=self.grid_size)
            if value is not None:
                self.history.append((row, col, self.editable_grid[row][col]))
                self.editable_grid[row][col] = value
                self.update_cell(row, col, value)

    def update_cell(self, row, col, value):
        cell, text = self.cells[row][col]
        self.canvas.itemconfig(text, text=str(value) if value != 0 else "")
        fill_color = "lightblue" if self.original_grid[row][col] != 0 else "white"
        self.canvas.itemconfig(cell, fill=fill_color)

    def create_control_panel(self):
        control_frame = ttk.Frame(self.right_frame)
        control_frame.pack(fill=X, expand=YES, pady=10)

        ttk.Label(control_frame, text="Độ khó:").pack(side=LEFT)
        self.difficulty_var = tk.StringVar(value="Dễ")
        difficulty_combo = ttk.Combobox(control_frame, textvariable=self.difficulty_var, 
                                        values=["Dễ", "Trung bình", "Khó", "Chuyên gia"])
        difficulty_combo.pack(side=LEFT, padx=(0, 10))

        ttk.Button(control_frame, text="Trò chơi mới", command=self.new_game).pack(side=LEFT)

        algorithm_frame = ttk.Frame(self.right_frame)
        algorithm_frame.pack(fill=X, expand=YES, pady=10)

        ttk.Label(algorithm_frame, text="Thuật toán:").pack(side=LEFT)
        self.algorithm_var = tk.StringVar(value="A*")
        algorithm_combo = ttk.Combobox(algorithm_frame, textvariable=self.algorithm_var, 
                                       values=["A*", "DFS", "BFS", "GA"])
        algorithm_combo.pack(side=LEFT, padx=(0, 10))

        self.solve_button = ttk.Button(algorithm_frame, text="Giải", command=self.solve) # Updated line
        self.solve_button.pack(side=LEFT)

        grid_size_frame = ttk.Frame(self.right_frame)
        grid_size_frame.pack(fill=X, expand=YES, pady=10)

        ttk.Label(grid_size_frame, text="Kích thước lưới:").pack(side=LEFT)
        self.grid_size_var = tk.StringVar(value="9x9")
        grid_size_combo = ttk.Combobox(grid_size_frame, textvariable=self.grid_size_var, 
                                       values=["9x9", "16x16"])
        grid_size_combo.pack(side=LEFT)
        grid_size_combo.bind("<<ComboboxSelected>>", self.change_grid_size)

        timer_frame = ttk.Frame(self.right_frame)
        timer_frame.pack(fill=X, expand=YES, pady=10)
        self.timer_label = ttk.Label(timer_frame, text="Thời gian: 00:00")
        self.timer_label.pack(side=LEFT)
        self.start_time = time.time()
        self.timer_running = False
        self.update_timer()

        undo_icon_image = Image.open("sudoku_images/icon.png").resize((20, 20), Image.LANCZOS)
        self.undo_icon = ImageTk.PhotoImage(undo_icon_image)

        # Nút Undo với icon
        undo_frame = ttk.Frame(self.right_frame)
        undo_frame.pack(fill=X, expand=YES, pady=10)
        undo_button = ttk.Button(undo_frame, text="Hoàn tác", image=self.undo_icon, compound=LEFT, command=self.undo_move,width=15)
        undo_button.pack(side=LEFT,padx=5)

        # Nút Mở Camera
        open_camera_button = ttk.Button(undo_frame, text="Mở Camera", command=self.open_camera, width=15)
        open_camera_button.pack(side=LEFT, padx=5)

        # Nút Đóng Camera
        close_camera_button = ttk.Button(undo_frame, text="Đóng Camera", command=self.close_camera, width=15)
        close_camera_button.pack(side=LEFT, padx=5)

      

        solution_frame = ttk.Frame(self.right_frame)
        solution_frame.pack(fill=X, expand=YES, pady=10)

        ttk.Button(self.right_frame, text="Tải lên hình ảnh", command=self.upload_image).pack(fill=X, expand=YES, pady=10)



        solution_button = ttk.Button(solution_frame, text="Kiểm tra giải pháp", command=self.check_solution, width=25)
        solution_button.pack(side=LEFT, expand=YES, padx=(0, 5))

        refresh_button = ttk.Button(solution_frame, text="Làm mới", command=self.refresh_grid, width=25)
        refresh_button.pack(side=LEFT, expand=YES, padx=(5, 0))

        self.info_label = ttk.Label(self.right_frame, text="")
        self.info_label.pack(fill=X, expand=YES, pady=10)
        # Thêm thah progress bar
        self.progress_bar = ttk.Progressbar(self.right_frame, orient=HORIZONTAL, length=200, mode='determinate', style="TProgressbar")
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
        self.solution_grid = None
        self.history = []
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
        
        self.update_timer()
        self.solution_displayed = False # Added line
        self.update_solve_button_text() # Added line




    def update_grid_display(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.update_cell(row, col, self.editable_grid[row][col])

    def solve(self):
        if not self.solution_displayed:
            self.timer_running = False
            self.progress_bar.stop()
            algorithm = self.algorithm_var.get()
            if algorithm == "GA":
                if self.grid_size == 9:
                    s = gss.Sudoku(9)
                    s.load(self.original_grid)
                    _, solution = s.solve()
                    self.solution_grid = solution.values if solution else None
                else:
                    s = gss.Sudoku(16)
                    s.load(self.original_grid)
                    _, solution = s.solve()
                    self.solution_grid = solution.values if solution else None

            elif algorithm == "DFS":
                    
                if self.grid_size == 9:
                    self.solution_grid = solveDFS(np.copy(self.original_grid))
                else:
                    self.solution_grid = solveIDS(np.copy(self.original_grid),16)

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
                


            if self.solution_grid is not None:
                self.info_label.config(text="Đang hiển thị giải pháp...")
                self.display_solution_gradually()
                self.solution_displayed = True

            else:
                self.info_label.config(text="Không tìm thấy giải pháp.")
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

    def change_grid_size(self, event):
        size = self.grid_size_var.get()
        self.grid_size = 9 if size == "9x9" else 16
        self.original_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.editable_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.solution_grid = None
        self.history = []
        self.draw_grid()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Tệp hình ảnh", "*.jpg;*.png;*.gif")])
        if file_path:
            sudoku_grid = extrapolate_sudoku(file_path, "models/model_sudoku_mnist.keras")
            self.original_grid = np.array(sudoku_grid)
            self.editable_grid = np.copy(self.original_grid)
            self.solution_grid = None
            self.history = []
            self.update_grid_display()
            if messagebox.askyesno("Xác nhận", "Đây có phải là câu đố Sudoku chính xác không?"):
                self.solve()
            else:
                messagebox.showinfo("Thử lại", "Vui lòng tải lên một hình ảnh khác.")
    def open_camera(self):
        self.cap = cv2.VideoCapture(0)  # Mở camera
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera.")
            return

        self.camera_window = tk.Toplevel(self)
        self.camera_window.title("Camera Real-Time Sudoku")

        self.camera_label = ttk.Label(self.camera_window)
        self.camera_label.pack()

        self.camera_running = True
        threading.Thread(target=self.update_camera_feed).start()

    def update_camera_feed(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                # Chuyển đổi khung hình sang dạng nhận diện số
                sudoku_grid, processed_frame = self.process_sudoku_frame(frame)

                if sudoku_grid is not None:
                    # Cập nhật lưới Sudoku từ kết quả nhận diện
                    self.original_grid = np.array(sudoku_grid)
                    self.editable_grid = np.copy(self.original_grid)
                    self.update_grid_display()

                # Hiển thị khung hình đã xử lý với các số nhận diện được
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = img_tk
                self.camera_label.configure(image=img_tk)

            self.camera_window.update_idletasks()
            self.camera_window.update()

    def process_sudoku_frame(self, frame):
        """
        Xử lý khung hình từ camera để nhận diện Sudoku và vẽ số lên khung hình.
        """
        try:
            # Nhận diện Sudoku từ khung hình
            sudoku_grid = extrapolate_sudoku(frame, "models/model_sudoku_mnist.keras")

            if sudoku_grid is not None:
                # Vẽ số nhận diện được lên khung hình
                for row in range(9):
                    for col in range(9):
                        number = sudoku_grid[row][col]
                        if number != 0:
                            x = int(col * frame.shape[1] / 9 + frame.shape[1] / 18)
                            y = int(row * frame.shape[0] / 9 + frame.shape[0] / 14)
                            cv2.putText(frame, str(number), (x, y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                return sudoku_grid, frame

            return None, frame

        except Exception as e:
            print(f"Lỗi xử lý khung hình: {e}")
            return None, frame

    def close_camera(self):
        self.camera_running = False
        self.cap.release()
        self.camera_window.destroy()

    def check_solution(self):
        if self.solution_grid is None:
            if not self.solution_displayed:
                self.timer_running = False
                self.progress_bar.stop()
                algorithm = self.algorithm_var.get()
                if algorithm == "GA":
                    if self.grid_size == 9:
                        s = gss.Sudoku(9)
                        s.load(self.original_grid)
                        _, solution = s.solve()
                        self.solution_grid = solution.values if solution else None
                    else:
                        s = gss.Sudoku(16)
                        s.load(self.original_grid)
                        _, solution = s.solve()
                        self.solution_grid = solution.values if solution else None

                elif algorithm == "DFS":
                        
                    if self.grid_size == 9:
                        self.solution_grid = solveDFS(np.copy(self.original_grid))
                    else:
                        self.solution_grid = solveIDS(np.copy(self.original_grid),16)

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
                    


                if self.solution_grid is not None:
                    self.info_label.config(text="Đang hiển thị giải pháp...")
                    # self.display_solution_gradually()
                    # self.solution_displayed = True

                else:
                    self.info_label.config(text="Không tìm thấy giải pháp.")
            else:
                self.hide_solution()
                self.solution_displayed = False
        
        self.update_solve_button_text()
    
        if self.solution_grid is not None:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    cell, _ = self.cells[row][col]
                    if self.original_grid[row][col] == 0:
                        if self.editable_grid[row][col] == self.solution_grid[row][col]:
                            self.canvas.itemconfig(cell, fill="lightgreen")
                        elif self.editable_grid[row][col] != 0:
                            self.canvas.itemconfig(cell, fill="lightcoral")
                        else:
                            self.canvas.itemconfig(cell, fill="white")
        else:
            messagebox.showwarning("Lỗi", "Không tìm thấy giải pháp hợp lệ.")


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
            self.progress_bar['value'] = (elapsed_time % 3600) / 36  # Reset every hour
            self.after(1000, self.update_timer)
    def refresh_grid(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.original_grid[row][col] == 0:
                    self.editable_grid[row][col] = 0
                    self.update_cell(row, col, 0)
        self.history = []
    def hide_solution(self): # Added method
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.original_grid[row][col] == 0:
                    self.editable_grid[row][col] = 0
                    self.update_cell(row, col, 0)
        self.info_label.config(text="Giải pháp đã được ẩn.")

    def update_solve_button_text(self): # Added method
        if self.solution_displayed:
            self.solve_button.config(text="Ẩn giải pháp")
        else:
            self.solve_button.config(text="Giải")
def run_app():
    app = SudokuApp("Crossover.json")
    app.mainloop()
if __name__ == "__main__":
    app = SudokuApp("Crossover.json")
    app.mainloop()

