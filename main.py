# # Import essential libraries 
# import requests 
# import cv2 
# import numpy as np 
# import imutils 

# # Replace the below URL with your own. Make sure to add "/shot.jpg" at last. 
# from PIL import Image, ImageTk
# import tkinter as tk


# import cv2
# import numpy as np
# import operator
# from keras.models import load_model
# from sudokuSolverDiversity import solveAStar, solveBFS, solveDFS,solveIDS,solveGreedy,is_valid_sudoku

# classifier = load_model("models/my_trained_model.keras")


# marge = 4
# case = 28 + 2 * marge
# taille_grille = 9 * case

# cap = cv2.VideoCapture("http://192.168.1.8:8080/video")
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# flag = 0
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1080, 620))


# while True:

#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (7, 7), 0)
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

#     contours, hierarchy = cv2.findContours(
#         thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contour_grille = None
#     maxArea = 0

#     for c in contours:
#         area = cv2.contourArea(c)
#         if area > 25000:
#             peri = cv2.arcLength(c, True)
#             polygone = cv2.approxPolyDP(c, 0.01 * peri, True)
#             if area > maxArea and len(polygone) == 4:
#                 contour_grille = polygone
#                 maxArea = area

#     if contour_grille is not None:
#         cv2.drawContours(frame, [contour_grille], 0, (0, 255, 0), 2)
#         points = np.vstack(contour_grille).squeeze()
#         points = sorted(points, key=operator.itemgetter(1))
#         if points[0][0] < points[1][0]:
#             if points[3][0] < points[2][0]:
#                 pts1 = np.float32([points[0], points[1], points[3], points[2]])
#             else:
#                 pts1 = np.float32([points[0], points[1], points[2], points[3]])
#         else:
#             if points[3][0] < points[2][0]:
#                 pts1 = np.float32([points[1], points[0], points[3], points[2]])
#             else:
#                 pts1 = np.float32([points[1], points[0], points[2], points[3]])
#         pts2 = np.float32([[0, 0], [taille_grille, 0], [0, taille_grille], [
#                           taille_grille, taille_grille]])
#         M = cv2.getPerspectiveTransform(pts1, pts2)
#         grille = cv2.warpPerspective(frame, M, (taille_grille, taille_grille))
#         grille = cv2.cvtColor(grille, cv2.COLOR_BGR2GRAY)
#         grille = cv2.adaptiveThreshold(
#             grille, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

#         cv2.imshow("grille", grille)
#         if flag == 0:

#             grille_txt = []
#             for y in range(9):
#                 ligne = ""
#                 for x in range(9):
#                     y2min = y * case + marge
#                     y2max = (y + 1) * case - marge
#                     x2min = x * case + marge
#                     x2max = (x + 1) * case - marge
#                     cv2.imwrite("mat" + str(y) + str(x) + ".png",
#                                 grille[y2min:y2max, x2min:x2max])
#                     img = grille[y2min:y2max, x2min:x2max]
#                     x = img.reshape(1, 28, 28, 1)
#                     if x.sum() > 10000:
#                         prediction = classifier.predict_classes(x)
#                         ligne += "{:d}".format(prediction[0])
#                     else:
#                         ligne += "{:d}".format(0)
#                 grille_txt.append(ligne)
#             print(grille_txt)
#             result = solveDFS(grille_txt)
#         print("Resultat:", result)

#         if result is not None:
#             flag = 1
#             fond = np.zeros(
#                 shape=(taille_grille, taille_grille, 3), dtype=np.float32)
#             for y in range(len(result)):
#                 for x in range(len(result[y])):
#                     if grille_txt[y][x] == "0":
#                         cv2.putText(fond, "{:d}".format(result[y][x]), ((
#                             x) * case + marge + 3, (y + 1) * case - marge - 3), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 0, 255), 1)
#             M = cv2.getPerspectiveTransform(pts2, pts1)
#             h, w, c = frame.shape
#             fondP = cv2.warpPerspective(fond, M, (w, h))
#             img2gray = cv2.cvtColor(fondP, cv2.COLOR_BGR2GRAY)
#             ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
#             mask = mask.astype('uint8')
#             mask_inv = cv2.bitwise_not(mask)
#             img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
#             img2_fg = cv2.bitwise_and(fondP, fondP, mask=mask).astype('uint8')
#             dst = cv2.add(img1_bg, img2_fg)
#             dst = cv2.resize(dst, (1080, 620))
#             cv2.imshow("frame", dst)
#             out.write(dst)

#         else:
#             frame = cv2.resize(frame, (1080, 620))
#             cv2.imshow("frame", frame)
#             out.write(frame)

#     else:
#         flag = 0
#         frame = cv2.resize(frame, (1080, 620))
#         cv2.imshow("frame", frame)
#         out.write(frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break


# out.release()
# cap.release()
# cv2.destroyAllWindows()

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import json
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image, ImageTk
import random
import time
from tkinter import filedialog, messagebox
import DiTruyen as gss
from sudokuSolverDiversity import solveAStar, solveBFS, solveDFS, solveIDS, solveGreedy
from ImageProcess import extrapolate_sudoku, displayImageSolution, resize_image
import math
import tkinter.simpledialog as simpledialog
import threading
from ttkbootstrap.tooltip import ToolTip

class SudokuApp(ttk.Window):
    def __init__(self, file):
        super().__init__(themename="darkly")
        self.title("Sudoku Solver")
        self.geometry("1200x800")

        self.grid_size = 9
        self.original_grid = np.zeros((9, 9), dtype=int)
        self.editable_grid = np.zeros((9, 9), dtype=int)
        self.solution_grid = None
        self.history = []
        self.solution_displayed = False
        self.hint_mode = False
        self.hinted_cells = []
        self.camera_running = False
        self.camera_index = 0
        self.max_retries = 3

        self.load_db(file)
        self.setup_ui()

    def load_db(self, file):
        with open(file) as f:
            data = json.load(f)
        self.easy = data['Easy']
        self.medium = data['Medium']
        self.hard = data['Hard']
        self.expert = data['Expert']

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
        self.canvas = tk.Canvas(self.left_frame, width=700, height=700, bg='white')
        self.canvas.pack(fill=BOTH, expand=YES)
        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        w, h = 700, 700
        padding_top = 40
        self.cell_size = w / self.grid_size

        for i in range(self.grid_size + 1):
            line_width = 4 if i % int(math.sqrt(self.grid_size)) == 0 else 2
            self.canvas.create_line(i * self.cell_size, padding_top, i * self.cell_size, h + padding_top, width=line_width)
            self.canvas.create_line(0, i * self.cell_size + padding_top, w, i * self.cell_size + padding_top, width=line_width)

        self.cells = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x, y = col * self.cell_size, row * self.cell_size + padding_top
                cell = self.canvas.create_rectangle(x, y, x + self.cell_size, y + self.cell_size, fill="white")
                text = self.canvas.create_text(x + self.cell_size/2, y + self.cell_size/2, text="", font=("Arial", 24))
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

        ttk.Label(control_frame, text="Độ khó:", font=("Arial", 14)).pack(side=LEFT)
        self.difficulty_var = tk.StringVar(value="Dễ")
        difficulty_combo = ttk.Combobox(control_frame, textvariable=self.difficulty_var, 
                                        values=["Dễ", "Trung bình", "Khó", "Chuyên gia"])
        difficulty_combo.pack(side=LEFT, padx=(0, 10))

        ttk.Button(control_frame, text="Trò chơi mới", command=self.new_game).pack(side=LEFT)

        algorithm_frame = ttk.Frame(self.right_frame)
        algorithm_frame.pack(fill=X, expand=YES, pady=10)

        ttk.Label(algorithm_frame, text="Thuật toán:", font=("Arial", 14)).pack(side=LEFT)
        self.algorithm_var = tk.StringVar(value="GA")
        algorithm_combo = ttk.Combobox(algorithm_frame, textvariable=self.algorithm_var, 
                                       values=["GA", "DFS", "BFS", "A*", "Greedy", "IDS"])
        algorithm_combo.pack(side=LEFT, padx=(0, 10))

        ttk.Button(algorithm_frame, text="Giải", command=self.solve).pack(side=LEFT)

        grid_size_frame = ttk.Frame(self.right_frame)
        grid_size_frame.pack(fill=X, expand=YES, pady=10)

        ttk.Label(grid_size_frame, text="Kích thước lưới:", font=("Arial", 14)).pack(side=LEFT)
        self.grid_size_var = tk.StringVar(value="9x9")
        grid_size_combo = ttk.Combobox(grid_size_frame, textvariable=self.grid_size_var, 
                                       values=["9x9", "16x16"])
        grid_size_combo.pack(side=LEFT)
        grid_size_combo.bind("<<ComboboxSelected>>", self.change_grid_size)

        timer_frame = ttk.Frame(self.right_frame)
        timer_frame.pack(fill=X, expand=YES, pady=10)
        self.timer_label = ttk.Label(timer_frame, text="Thời gian: 00:00", font=("Arial", 16))
        self.timer_label.pack(side=LEFT)
        self.start_time = time.time()
        self.timer_running = False
        self.update_timer()

        undo_frame = ttk.Frame(self.right_frame)
        undo_frame.pack(fill=X, expand=YES, pady=10)
        
        # Load icons
        undo_icon_image = Image.open("sudoku_images/icon.png").resize((30, 30), Image.LANCZOS)
        self.undo_icon = ImageTk.PhotoImage(undo_icon_image)
        refresh_icon_image = Image.open("Repository/refresh_icon.png").resize((30, 30), Image.LANCZOS)
        self.refresh_icon = ImageTk.PhotoImage(refresh_icon_image)
        open_icon_image = Image.open("Repository/open.png").resize((30, 30), Image.LANCZOS)
        self.open_icon = ImageTk.PhotoImage(open_icon_image)
        close_icon_image = Image.open("Repository/close.png").resize((30, 30), Image.LANCZOS)
        self.close_icon = ImageTk.PhotoImage(close_icon_image)
        hint_icon_image = Image.open("Repository/hint_icon.png").resize((30, 30), Image.LANCZOS)
        self.hint_icon = ImageTk.PhotoImage(hint_icon_image)
        upload_icon_image = Image.open("Repository/camera.png").resize((30, 30), Image.LANCZOS)
        self.upload_icon = ImageTk.PhotoImage(upload_icon_image)
        check_icon_image = Image.open("Repository/check_icon.png").resize((30, 30), Image.LANCZOS)
        self.check_icon = ImageTk.PhotoImage(check_icon_image)

        undo_button = ttk.Button(undo_frame, text="Hoàn tác", image=self.undo_icon, compound=LEFT, command=self.undo_move, width=20, cursor="hand2")
        undo_button.pack(side=LEFT)
        self.open_camera_button = ttk.Button(undo_frame, text="Open Camera", image=self.open_icon, command=self.show_camera_menu, width=20, cursor="hand2")
        self.open_camera_button.pack(side=LEFT, padx=5)
        ToolTip(self.open_camera_button, text="Open Camera")
        self.ip_camera_entry = ttk.Entry(undo_frame, width=15)
        self.ip_camera_entry.insert(0, "http://192.168.1.100:8080/video")
        self.ip_camera_entry.pack(side=LEFT, padx=5)
        close_camera_button = ttk.Button(undo_frame, text="Đóng Camera", image=self.close_icon, command=self.close_camera, width=20, cursor="hand2")
        close_camera_button.pack(side=LEFT, padx=5)
        hint_button = ttk.Button(undo_frame, text="Gợi ý", image=self.hint_icon, compound=LEFT, command=self.toggle_hint_mode, width=20, cursor="hand2")
        hint_button.pack(side=LEFT, padx=5)


        refresh_button = ttk.Button(undo_frame, text="Làm mới", image=self.refresh_icon, compound=LEFT, command=self.refresh_grid, width=20, cursor="hand2")
        refresh_button.pack(side=LEFT, padx=5)

        ttk.Button(self.right_frame, text="Tải lên hình ảnh", image=self.upload_icon, compound=LEFT, command=self.upload_image).pack(fill=X, expand=YES, pady=10)
        

        solution_frame = ttk.Frame(self.right_frame)
        solution_frame.pack(fill=X, expand=YES, pady=10)

        solution_button = ttk.Button(solution_frame, text="Kiểm tra đáp án", image=self.check_icon, compound=LEFT, command=self.check_solution, width=30, cursor="hand2")
        solution_button.pack(side=LEFT, expand=YES, padx=(0, 5))

        refresh_button = ttk.Button(solution_frame, text="Làm mới", image=self.refresh_icon, compound=LEFT, command=self.refresh_grid, width=30, cursor="hand2")
        refresh_button.pack(side=LEFT, expand=YES, padx=(5, 0))


        self.info_label = ttk.Label(self.right_frame, text="", font=("Arial", 14))
        self.info_label.pack(fill=X, expand=YES, pady=10)
        
        # Add progress bar with improved appearance
        self.progress_bar = ttk.Progressbar(self.right_frame, orient=HORIZONTAL, length=300, mode='determinate', style="TProgressbar")
        self.progress_bar.pack(fill=X, expand=YES, pady=10)
        
        # Configure the progress bar style
        style = ttk.Style()
        style.configure("TProgressbar", thickness=20, troughcolor='#E0E0E0', background='#4CAF50')


    def new_game(self):
        level = self.difficulty_var.get()
        if level == "Dễ":
            puzzle = random.choice(self.easy)
        elif level == "Trung bình":
            puzzle = random.choice(self.medium)
        elif level == "Khó":
            puzzle = random.choice(self.hard)
        elif level == "Chuyên gia":
            puzzle = random.choice(self.expert)
        else:
            puzzle = "0" * 81  

        puzzle = list(map(int, puzzle))
        self.original_grid = np.array(puzzle).reshape((9, 9))
        self.editable_grid = np.copy(self.original_grid)
        self.solution_grid = None
        self.history = []
        self.update_grid_display()
        
        self.timer_running = False
        self.start_time = time.time()
        self.timer_label.config(text="Thời gian: 00:00")
        self.timer_running = True
        
        self.progress_bar.stop()
        self.progress_bar['value'] = 0
        self.progress_bar.start()
        
        self.update_timer()

    def update_grid_display(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.update_cell(row, col, self.editable_grid[row][col])

    def solve(self):
        self.timer_running = False
        self.progress_bar.stop()
        algorithm = self.algorithm_var.get()
        if algorithm == "GA":
            s = gss.Sudoku()
            s.load(self.original_grid)
            _, solution = s.solve()
            self.solution_grid = solution.values if solution else None
        elif algorithm == "DFS":
            self.solution_grid = solveDFS(np.copy(self.original_grid))
        elif algorithm == "BFS":
            self.solution_grid = solveBFS(np.copy(self.original_grid))
        elif algorithm == "A*":
            self.solution_grid = solveAStar(np.copy(self.original_grid))
        elif algorithm == "Greedy":
            self.solution_grid = solveGreedy(np.copy(self.original_grid))
        elif algorithm == "IDS":
            self.solution_grid = solveIDS(np.copy(self.original_grid))

        if self.solution_grid is not None:
            self.info_label.config(text="Đang hiển thị giải pháp...")
            self.display_solution_gradually()
        else:
            self.info_label.config(text="Không tìm thấy giải pháp.")

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
            self.after(50, self.display_next_solution_cell)
        else:
            self.info_label.config(text="Câu đố đã được giải thành công!")
            self.solution_displayed = True


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
            sudoku_grid, _ = extrapolate_sudoku(file_path, "models/model_sudoku_mnist.keras")
            self.original_grid = np.array(sudoku_grid)
            self.editable_grid = np.copy(self.original_grid)
            self.solution_grid = None
            self.history = []
            self.update_grid_display()

            if messagebox.askyesno("Xác nhận", "Đây có phải là câu đố Sudoku chính xác không?"):
                self.solve()
                if self.solution_grid is not None:
                    result_image = displayImageSolution(file_path, self.solution_grid, self.original_grid)
                    result_image = resize_image(result_image)
                    cv2.imshow("Sudoku Solved", result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    messagebox.showinfo("Thông báo", "Không thể tìm được lời giải cho Sudoku này.")
            else:
                messagebox.showinfo("Thử lại", "Vui lòng tải lên một hình ảnh khác.")

    def check_solution(self):
        if self.solution_grid is None:
            self.solve()
        
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
            self.progress_bar['value'] = (elapsed_time % 3600) / 36
            self.after(1000, self.update_timer)

    def refresh_grid(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.original_grid[row][col] == 0:
                    self.editable_grid[row][col] = 0
                    self.update_cell(row, col, 0)
        self.history = []

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera.")
            return

        self.camera_window = tk.Toplevel(self)
        self.camera_window.title("Camera Real-Time Sudoku")
        self.camera_window.geometry("800x600")

        self.camera_frame = ttk.Frame(self.camera_window)
        self.camera_frame.pack(fill=tk.BOTH, expand=tk.YES)

        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=tk.YES)

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

        return frame

    def close_camera(self):
        self.camera_running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'camera_window'):
            self.camera_window.destroy()

    def toggle_hint_mode(self):
        self.hint_mode = not self.hint_mode
        if self.hint_mode:
            self.give_hint()

    def give_hint(self):
        # Logic to provide a hint (implementation needed)
        pass

    def show_camera_menu(self):
        camera_menu = tk.Menu(self, tearoff=0)
        camera_menu.add_command(label="Open Webcam", command=self.open_webcam)
        camera_menu.add_command(label="Open Phone Camera", command=self.open_phone_camera)
        
        # Get the position of the "Open Camera" button
        button_x = self.open_camera_button.winfo_rootx()
        button_y = self.open_camera_button.winfo_rooty() + self.open_camera_button.winfo_height()
        
        # Display the menu at the button's position
        camera_menu.tk_popup(button_x, button_y)

    def open_webcam(self):
        # Existing webcam opening logic
        cap = cv2.VideoCapture(0)
        self.start_camera_feed(cap)

    def open_phone_camera(self):
        # Phone camera opening logic
        ip_camera_url = "http://192.168.1.4:8080/video"  # Update with your phone's IP camera URL
        cap = cv2.VideoCapture(ip_camera_url)
        self.start_camera_feed(cap)

    def start_camera_feed(self, cap):
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to connect to the camera.")
            return

        self.camera_window = tk.Toplevel(self)
        self.camera_window.title("Real-Time Sudoku Camera")
        self.camera_window.geometry("800x600")

        self.camera_frame = ttk.Frame(self.camera_window)
        self.camera_frame.pack(fill=tk.BOTH, expand=tk.YES)

        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=tk.YES)

        self.camera_running = True
        threading.Thread(target=self.update_camera_feed_myself, args=(cap,), daemon=True).start()


if __name__ == "__main__":
    app = SudokuApp("Crossover.json")
    app.mainloop()

