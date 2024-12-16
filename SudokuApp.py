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

from sudokuSolverDiversity import solveAStar, solveBFS, solveDFS,solveIDS,solveGreedy
from ImageProcess import extrapolate_sudoku,displayImageSolution,resize_image
import math
import tkinter.simpledialog as simpledialog
import threading  # Để chạy camera trong một luồng riêng
from DiTruyen import solveGA   # Import the genetic solver
from ttkbootstrap.tooltip import ToolTip
class SudokuApp(ttk.Toplevel):
    def __init__(self, file):
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
        self.camera_index = 0
        self.max_retries = 3
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
                                       values=["A*", "DFS", "BFS", "GA","Greedy","IDS"])
        algorithm_combo.pack(side=LEFT, padx=(0, 10))

        self.solve_button = ttk.Button(algorithm_frame, text="Giải", command=self.solve,cursor="hand2", width=30) # Updated line
        self.solve_button.pack(side=LEFT)

        grid_size_frame = ttk.Frame(self.right_frame)
        grid_size_frame.pack(fill=X, expand=YES, pady=10)

        ttk.Label(grid_size_frame, text="Kích thước lưới:", font=("Roboto", 14)).pack(side=LEFT) # Updated font size

        self.grid_size_var = tk.StringVar(value="9x9")
        grid_size_combo = ttk.Combobox(grid_size_frame, textvariable=self.grid_size_var, 
                                       values=["9x9", "16x16"])
        grid_size_combo.pack(side=LEFT)
        grid_size_combo.bind("<<ComboboxSelected>>", self.change_grid_size)

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
        open_camera_button = ttk.Button(undo_frame, text="Mở Camera",image=self.open_icon, command=self.open_camera, width=20,cursor="hand2")
        open_camera_button.pack(side=LEFT, padx=5)
        ToolTip(open_camera_button, text="Mở Camera") # Thêm tooltip


        # Nút Đóng Camera
        close_camera_button = ttk.Button(undo_frame, text="Đóng Camera", image=self.close_icon,command=self.close_camera, width=20,cursor="hand2")
        close_camera_button.pack(side=LEFT, padx=5)
        ToolTip(close_camera_button, text="Tắt Camera") # Thêm tooltip




        hint_button = ttk.Button(undo_frame, text="Gợi ý", image=self.hint_icon, compound=LEFT, command=self.toggle_hint_mode, width=20, cursor="hand2")
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
        
        self.update_timer()
        self.solution_displayed = False # Added line
        self.update_solve_button_text() # Added line






    def update_grid_display(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.update_cell(row, col, self.editable_grid[row][col])

    def solve(self):
        if np.all(self.original_grid == 0):
            messagebox.showwarning("Lỗi", "Vui lòng tạo một trò chơi mới trước khi giải.")
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
                
            elif algorithm == "Greedy":
                if self.grid_size == 9:

                    self.solution_grid = solveGreedy(np.copy(self.original_grid))
                else:
                    self.solution_grid = solveGreedy(np.copy(self.original_grid),16)
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

    def change_grid_size(self, event):
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
        if file_path:
            # Giải Sudoku từ ảnh
            sudoku_grid, largest_rect_coord, transform_matrix, warp_size = extrapolate_sudoku(file_path, "models/model_sudoku_mnist.keras")

            self.original_grid = np.array(sudoku_grid)
            self.editable_grid = np.copy(self.original_grid)
            self.solution_grid = None
            self.history = []
            self.update_grid_display()

            # Xác nhận câu đố Sudoku
            if messagebox.askyesno("Xác nhận", "Đây có phải là câu đố Sudoku chính xác không?"):
                self.solve()
                # Hiển thị ảnh với lời giải


                result_image = displayImageSolution(file_path, self.solution_grid, self.original_grid,largest_rect_coord)

                result_image = resize_image(result_image)  # Resize the image if it's too large

                cv2.imshow("Sudoku Solved", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                messagebox.showinfo("Thử lại", "Vui lòng tải lên một hình ảnh khác.")


    def open_camera(self):
        for i in range(self.max_retries):
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    break
            self.cap.release()
            self.camera_index += 1
    
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", f"Không thể mở camera sau {self.max_retries} lần thử.")
            return

        self.camera_window = tk.Toplevel(self)
        self.camera_window.title("Camera Real-Time Sudoku")
        self.camera_window.geometry("800x600")

        self.camera_frame = ttk.Frame(self.camera_window)
        self.camera_frame.pack(fill=BOTH, expand=YES)

        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=BOTH, expand=YES)

        self.camera_running = True
        threading.Thread(target=self.update_camera_feed, daemon=True).start()

    def update_camera_feed(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Lỗi", "Không thể đọc khung hình từ camera.")
                self.close_camera()
                break

            sudoku_grid, processed_frame, largest_rect_coord = self.process_sudoku_frame(frame)

            if sudoku_grid is not None:
                self.original_grid = sudoku_grid
                self.editable_grid = np.copy(self.original_grid)
                self.update_grid_display()
                processed_frame = self.solve_and_draw_solution(processed_frame, sudoku_grid, largest_rect_coord)
            else:
                processed_frame = frame

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (800, 600))
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = img_tk
            self.camera_label.configure(image=img_tk)

            self.camera_window.update_idletasks()
            self.camera_window.update()

    def process_sudoku_frame(self, frame):
        sudoku_grid, largest_rect_coord = extrapolate_sudoku(frame, "models/model_sudoku_mnist.keras")
        return sudoku_grid, frame, largest_rect_coord

    def solve_and_draw_solution(self, frame, sudoku_grid, largest_rect_coord):
        algorithm = self.algorithm_var.get()
        if algorithm == "GA":
            self.solution_grid = solveGreedy(np.copy(sudoku_grid))
        elif algorithm == "DFS":
            self.solution_grid = solveDFS(np.copy(sudoku_grid))
        elif algorithm == "BFS":
            self.solution_grid = solveBFS(np.copy(sudoku_grid))
        elif algorithm == "A*":
            self.solution_grid = solveAStar(np.copy(sudoku_grid))
        elif algorithm == "Greedy":
            self.solution_grid = solveGreedy(np.copy(sudoku_grid))
        else:
            self.solution_grid = solveIDS(np.copy(sudoku_grid))

        if self.solution_grid is not None:
            height, width = frame.shape[:2]
            cell_height = height // 9
            cell_width = width // 9

            # Draw the green rectangle around the Sudoku grid
            if largest_rect_coord is not None and len(largest_rect_coord) == 4:
                cv2.polylines(frame, [largest_rect_coord], True, (0, 255, 0), 3)

            for row in range(9):
                for col in range(9):
                    if sudoku_grid[row][col] == 0:
                        number = self.solution_grid[row][col]
                        x = int(col * cell_width + cell_width // 2)
                        y = int(row * cell_height + cell_height // 2)
                        cv2.putText(frame, str(number), (x, y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for i in range(10):
                thickness = 3 if i % 3 == 0 else 1
                cv2.line(frame, (i * cell_width, 0), (i * cell_width, height), (0, 255, 0), thickness)
                cv2.line(frame, (0, i * cell_height), (width, i * cell_height), (0, 255, 0), thickness)

    def close_camera(self):
        self.camera_running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'camera_window'):
            self.camera_window.destroy()

    def process_sudoku_frame2(self, frame):
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


    def check_solution(self):
        if np.all(self.original_grid == 0):
            messagebox.showwarning("Lỗi", "Vui lòng tạo một trò chơi mới trước khi giải.")
            return
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
                    
                elif algorithm == "Greedy":
                    if self.grid_size == 9:

                        self.solution_grid = solveGreedy(np.copy(self.original_grid))
                    else:
                        self.solution_grid = solveGreedy(np.copy(self.original_grid),16)
                else:
                    if self.grid_size == 9:
                        self.solution_grid = solveIDS(np.copy(self.original_grid))
                    else:
                        self.solution_grid = solveIDS(np.copy(self.original_grid),16)
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
                messagebox.showwarning("Lỗi", "Không tìm thấy đáp án hợp lệ.")


                            


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
            self.progress_bar['value'] = (elapsed_time % 3600) / 36  # Reset every hour
            self.after(1000, self.update_timer)
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


def run_app():
    app = SudokuApp("Crossover.json")
    app.mainloop()
if __name__ == "__main__":
    app = SudokuApp("Crossover.json")
    app.mainloop()

