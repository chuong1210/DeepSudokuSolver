import heapq
import numpy as np
from collections import deque
from copy import deepcopy
import heapq
import numpy as np
from collections import deque
from copy import deepcopy

def solveDFS(board, grid_size=9):
    """
    Giải Sudoku bằng thuật toán Depth-First Search (DFS).
    Trả về bảng Sudoku đã được giải hoặc None nếu không có lời giải.
    """
    print("DFS SOLVE")

    def is_valid(board, row, col, num):
        # Kiểm tra hàng
        for i in range(grid_size):
            if board[row][i] == num:
                return False

        # Kiểm tra cột
        for i in range(grid_size):
            if board[i][col] == num:
                return False

        # Kiểm tra vùng box (subgrid)
        box_size = int(grid_size ** 0.5)  # Lấy kích thước của box (3x3 với 9x9, 4x4 với 16x16)
        box_row = row // box_size * box_size
        box_col = col // box_size * box_size
        for i in range(box_row, box_row + box_size):
            for j in range(box_col, box_col + box_size):
                if board[i][j] == num:
                    return False
        return True

    def solve(board):
        for row in range(grid_size):
            for col in range(grid_size):
                if board[row][col] == 0:
                    for num in range(1, grid_size + 1):
                        if is_valid(board, row, col, num):
                            board[row][col] = num
                            if solve(board):
                                return True
                            board[row][col] = 0  # Quay lại
                    return False
        return True  # Nếu tất cả các ô đã được điền

    board_copy = [row[:] for row in board]  # Sao chép bảng để tránh thay đổi bảng gốc
    if solve(board_copy):
        return board_copy
    else:
        return None





def solveBFS(board, grid_size=9):
    print("BFS SOLVE")

    def findEmpty(board):
        for i in range(grid_size):
            for j in range(grid_size):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def valid(board, num, pos):
        # Kiểm tra cột
        for i in range(grid_size):
            if board[pos[0]][i] == num and pos[1] != i:
                return False
        # Kiểm tra hàng
        for j in range(grid_size):
            if board[j][pos[1]] == num and pos[0] != j:
                return False
        # Kiểm tra ô box
        box_size = int(grid_size ** 0.5)
        row_x = pos[1] // box_size
        row_y = pos[0] // box_size
        for i in range(row_y * box_size, row_y * box_size + box_size):
            for j in range(row_x * box_size, row_x * box_size + box_size):
                if board[i][j] == num and (i, j) != pos:
                    return False
        return True

    queue = deque([board])  # Initialize the BFS queue with the initial board

    while queue:
        current_board = queue.popleft()  # Get the next board state from the queue

        if all(current_board[row][col] != 0 for row in range(grid_size) for col in range(grid_size)):
            return current_board  # Return the solved board

        find = findEmpty(current_board)
        if not find:
            continue

        row, col = find

        for num in range(1, grid_size + 1):
            if valid(current_board, num, (row, col)):  # Kiểm tra tính hợp lệ
                print(current_board)

                new_board = deepcopy(current_board)
                new_board[row][col] = num  # Điền số vào ô trống
                queue.append(new_board)

    return None


def solveAStar(board, grid_size=9):
        """
        Giải câu đố Sudoku bằng giải thuật A*.
        Trả về bảng Sudoku đã được giải hoặc `None` nếu không có lời giải.
        """

        print("A* SOLVE")

        # Đảm bảo bảng đầu vào là danh sách Python (không phải numpy array)
        if isinstance(board, np.ndarray):
            board = board.tolist()

        # Hàm tìm ô trống đầu tiên trên bảng
        def find_empty(board):
            for i in range(grid_size):  # Duyệt qua từng hàng
                for j in range(grid_size):  # Duyệt qua từng cột
                    if board[i][j] == 0:  # Nếu ô trống (giá trị 0)
                        return i, j  # Trả về vị trí (hàng, cột)
            return None  # Không còn ô trống nào

        # Hàm kiểm tra xem số `num` có hợp lệ tại vị trí `pos` hay không
        def is_valid(board, num, pos):
            row, col = pos

            # Kiểm tra hàng (row)
            if num in board[row]:
                return False

            # Kiểm tra cột (column)
            for i in range(grid_size):
                if board[i][col] == num:
                    return False

            # Kiểm tra ô 3x3 (subgrid)
            box_size = int(grid_size ** 0.5)
            box_row = row // box_size * box_size
            box_col = col // box_size * box_size
            for i in range(box_row, box_row + box_size):
                for j in range(box_col, box_col + box_size):
                    if board[i][j] == num:
                        return False

            return True  # Nếu không vi phạm quy tắc nào, trả về hợp lệ

        # Hàm heuristic: Đếm số lượng ô trống trên bảng
        def heuristic(board):
            return sum(row.count(0) for row in board)  # Dùng `count(0)` để đếm số 0 trong mỗi hàng

        # Hàm sinh các trạng thái kế tiếp (neighbor) bằng cách điền vào một ô trống
        def generate_neighbors(board):
            neighbors = []
            empty_pos = find_empty(board)  # Tìm ô trống đầu tiên

            if empty_pos is None:  # Nếu không còn ô trống nào, trả về danh sách rỗng
                return []

            row, col = empty_pos  # Lấy vị trí (hàng, cột) của ô trống

            for num in range(1, grid_size+1):  # Thử tất cả các số từ 1 đến 9
                if is_valid(board, num, (row, col)):  # Kiểm tra số `num` có hợp lệ
                    print(board)
                    new_board = [row[:] for row in board]  # Tạo bản sao của bảng hiện tại
                    new_board[row][col] = num  # Điền số vào ô trống
                    neighbors.append(new_board)  # Thêm trạng thái mới vào danh sách

            return neighbors

        # Hàng đợi ưu tiên cho giải thuật A*
        open_list = []  # Dùng heapq để quản lý hàng đợi ưu tiên
        visited = set()  # Tập hợp để lưu các trạng thái đã duyệt (tránh lặp lại)

        # Thêm trạng thái ban đầu vào hàng đợi với chi phí heuristic
        heapq.heappush(open_list, (heuristic(board), 0, board))  # (chi phí heuristic, chi phí đường đi, bảng)

        while open_list:
            # Lấy bảng có giá trị f = g(n) + h(n) nhỏ nhất
            _, g, current_board = heapq.heappop(open_list)

            # Kiểm tra nếu bảng đã được giải (không còn ô trống)
            if find_empty(current_board) is None:
                return current_board  # Trả về bảng đã giải

            # Thêm trạng thái hiện tại vào tập đã duyệt
            visited.add(tuple(map(tuple, current_board)))  # Chuyển bảng sang tuple để làm khóa (immutable)

            # Sinh các trạng thái kế tiếp (neighbors)
            for neighbor in generate_neighbors(current_board):
                neighbor_tuple = tuple(map(tuple, neighbor))  # Chuyển sang tuple để so sánh

                if neighbor_tuple not in visited:  # Nếu trạng thái chưa được duyệt
                    f = g + 1 + heuristic(neighbor)  # f(n) = g(n) + h(n)
                    heapq.heappush(open_list, (f, g + 1, neighbor))  # Thêm vào hàng đợi

        return None  # Nếu không tìm thấy lời giải, trả về None





def solveGreedy(board, grid_size=9):
    """
    Giải Sudoku bằng thuật toán Greedy.
    - Thử điền từng số vào ô trống và kiểm tra tính hợp lệ.
    - Nếu số hợp lệ, tiếp tục điền vào ô tiếp theo.
    - Nếu không thể điền được số hợp lệ, quay lại ô trước đó và thử số khác.
    - Trả về bảng đã giải hoặc None nếu không có lời giải.
    """
    print("GREEDY SOLVE")
    box_size = int(grid_size**0.5)

    def is_valid(row, col, num):
        """Kiểm tra xem số `num` có hợp lệ tại ô (row, col) hay không."""
        # Kiểm tra hàng
        for x in range(grid_size):
            if board[row][x] == num:
                return False

        # Kiểm tra cột
        for x in range(grid_size):
            if board[x][col] == num:
                return False

        # Kiểm tra ô vuông con
        start_row = row - row % box_size
        start_col = col - col % box_size
        for i in range(box_size):
            for j in range(box_size):
                if board[i + start_row][j + start_col] == num:
                    return False

        return True


    def find_empty_location(l):
        """Tìm ô trống tiếp theo."""
        for row in range(grid_size):
            for col in range(grid_size):
                if board[row][col] == 0:
                    l[0] = row
                    l[1] = col
                    return True
        return False

    def solve():
        """Hàm đệ quy để giải Sudoku."""
        l = [0, 0]

        if not find_empty_location(l):
            return True  # Không còn ô trống, đã giải xong

        row = l[0]
        col = l[1]

        for num in range(1, grid_size + 1):  # Thử các số từ 1 đến grid_size
            if is_valid(row, col, num):
                board[row][col] = num

                if solve():
                    return True

                board[row][col] = 0  # Quay lui nếu không tìm thấy lời giải

        return False

    if solve():
        return board
    else:
        return None


# Hàm IDS giải Sudoku
def solveIDS(board, size=9):
    print("IDS SOLVE")
    # Hàm tìm ô trống đầu tiên
    def find_empty(board, size):
        for row in range(size):
            for col in range(size):
                if board[row][col] == 0:
                    return row, col
        return None
    # Hàm kiểm tra xem có thể điền số num vào vị trí (row, col) hay không
    def is_valid(board, row, col, num, size):
        # Kiểm tra trong hàng
        if num in board[row]:
            return False

        # Kiểm tra trong cột
        if num in board[:, col]:
            return False

        # Kiểm tra trong khối nhỏ (3x3 cho 9x9 và 4x4 cho 16x16)
        box_size = int(size ** 0.5)
        start_row, start_col = box_size * (row // box_size), box_size * (col // box_size)
        if num in board[start_row:start_row + box_size, start_col:start_col + box_size]:
            return False

        return True




    # Hàm tìm kiếm theo chiều sâu với giới hạn độ sâu (DLS)
    def depth_limited_search(board, size, limit):
        empty = find_empty(board, size)
        if not empty:
            return True  # Không còn ô trống, Sudoku đã được giải

        if limit <= 0:
            return False  # Vượt quá giới hạn độ sâu

        row, col = empty

        for num in range(1, size + 1):
            if is_valid(board, row, col, num, size):
                board[row][col] = num
                if depth_limited_search(board, size, limit - 1):
                    return True
                board[row][col] = 0  # Quay lui nếu không giải được

        return None
    depth_limit = 1
    while True:
        if depth_limited_search(board, size, depth_limit):
            return board
        depth_limit += 1

def is_valid_sudoku(grid):
    # Kiểm tra nếu lưới Sudoku chỉ chứa toàn số 0
    if all(cell == 0 for row in grid for cell in row):
        return False

    # Kiểm tra từng hàng
    for row in grid:
        if not is_valid_group(row):
            return False

    # Kiểm tra từng cột
    for col in zip(*grid):
        if not is_valid_group(col):
            return False

    # Kiểm tra từng ô vuông 3x3
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = [grid[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            if not is_valid_group(block):
                return False

    return True

def is_valid_group(group):
    # Loại bỏ các số 0 (ô trống) và kiểm tra trùng lặp
    nums = [num for num in group if num != 0]
    return len(nums) == len(set(nums))
def allPositionsFilled(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True