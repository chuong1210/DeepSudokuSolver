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


def iterative_deepening_search(board, grid_size=9):
    max_depth = 1
    while True:
        if solveDFS(board, grid_size):
            return board  # Nếu tìm được giải pháp, trả về bảng đã giải
        max_depth += 1  # Tăng độ sâu và thử lại

def greedy_sudoku_solver(board, grid_size=9):
    def is_valid(board, row, col, num):
        # Kiểm tra hàng
        if num in board[row]:
            return False
        # Kiểm tra cột
        if num in board[:, col]:
            return False
        # Kiểm tra khối box
        box_size = int(grid_size ** 0.5)
        start_row, start_col = box_size * (row // box_size), box_size * (col // box_size)
        for i in range(start_row, start_row + box_size):
            for j in range(start_col, start_col + box_size):
                if board[i][j] == num:
                    return False
        return True

    for row in range(grid_size):
        for col in range(grid_size):
            if board[row][col] == 0:  # Nếu ô trống
                for num in range(1, grid_size + 1):
                    if is_valid(board, row, col, num):
                        board[row][col] = num  # Điền số vào ô trống
                        if greedy_sudoku_solver(board, grid_size):
                            return True
                        board[row][col] = 0
                return False  # Nếu không thể điền vào ô này, trả về False
    return True  # Nếu không còn ô trống, tức là đã giải được


def solveBFS(board, grid_size=9):
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
                new_board = deepcopy(current_board)
                new_board[row][col] = num  # Điền số vào ô trống
                queue.append(new_board)

    return None


def solveAStar(board, grid_size=9):
        """
        Giải câu đố Sudoku bằng giải thuật A*.
        Trả về bảng Sudoku đã được giải hoặc `None` nếu không có lời giải.
        """

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




def solveIDS(board, grid_size):
    # Hàm kiểm tra xem một số có hợp lệ tại vị trí (row, col) trên bảng hay không
    def is_valid(board, row, col, num, grid_size):
        box_size = int(grid_size ** 0.5)  # Tính kích thước của ô vuông con (box)

        # Kiểm tra hàng (row)
        for i in range(grid_size):
            if board[row][i] == num:
                return False

        # Kiểm tra cột (col)
        for i in range(grid_size):
            if board[i][col] == num:
                return False

        # Kiểm tra vùng box (ô vuông con)
        box_row = row // box_size * box_size
        box_col = col // box_size * box_size
        for i in range(box_row, box_row + box_size):
            for j in range(box_col, box_col + box_size):
                if board[i][j] == num:
                    return False
        return True

    # Hàm DFS với giới hạn độ sâu
    def dfs_depth_limit(board, depth_limit, grid_size):
        # Hàm DFS đệ quy, nhận vào độ sâu hiện tại
        def dfs(board, depth):
            if depth > depth_limit:
                return None  # Nếu vượt quá độ sâu giới hạn thì trả về None

            # Kiểm tra xem bảng đã đầy chưa (tức là tất cả các ô đã được điền số)
            for row in range(grid_size):
                for col in range(grid_size):
                    if board[row][col] == 0:  # Nếu tìm thấy ô trống
                        for num in range(1, grid_size + 1):  # Thử các số từ 1 đến grid_size
                            if is_valid(board, row, col, num, grid_size):  # Kiểm tra tính hợp lệ
                                board[row][col] = num
                                result = dfs(board, depth + 1)  # Gọi đệ quy để điền tiếp
                                if result is not None:
                                    return result  # Nếu tìm thấy kết quả thì trả về
                                board[row][col] = 0  # Quay lại nếu không tìm thấy
                        return None  # Nếu không có số nào hợp lệ thì quay lại
            return board  # Nếu không còn ô trống, tức là đã giải xong

        return dfs(board, 0)

    # Thực hiện IDS: Bắt đầu từ độ sâu 1 và tăng dần độ sâu
    for depth_limit in range(1, 100):  # Có thể điều chỉnh giới hạn độ sâu
        board_copy = [row[:] for row in board]  # Sao chép bảng để không làm thay đổi bảng gốc
        result = dfs_depth_limit(board_copy, depth_limit, grid_size)  # Giải bài toán với độ sâu hiện tại
        if result is not None:
            return result  # Nếu tìm thấy lời giải thì trả về

    return None  # Nếu không tìm thấy lời giải trong giới hạn độ sâu

def solveGreedy(board, grid_size):
    """
    Giải Sudoku bằng thuật toán Greedy.
    - Thử điền từng số vào ô trống và kiểm tra tính hợp lệ.
    - Nếu số hợp lệ, tiếp tục điền vào ô tiếp theo.
    - Nếu không thể điền được số hợp lệ, quay lại ô trước đó và thử số khác.
    - Trả về bảng đã giải hoặc None nếu không có lời giải.
    """
    def is_valid(board, row, col, num, grid_size):
        box_size = int(grid_size ** 0.5)  # Tính kích thước của ô vuông con (box)

        # Kiểm tra trong cùng một hàng
        if num in board[row]:
            return False
         # Kiểm tra trong cùng một cột
        # if num in board[:, col]:

        if num in [board[i][col] for i in range(grid_size)]:
            return False
        start_row, start_col = box_size * (row // box_size), box_size * (col // box_size)
        for i in range(start_row, start_row + box_size):
            for j in range(start_col, start_col + box_size):
                if board[i][j] == num:
                    return False
        return True

    for row in range(grid_size):
        for col in range(grid_size):
            if board[row][col] == 0:  # Nếu ô trống
                for num in range(1, grid_size + 1):  # Duyệt thử các số từ 1 đến grid_size
                    if is_valid(board, row, col, num, grid_size):  # Kiểm tra số có hợp lệ không
                        board[row][col] = num
                        if solveGreedy(board, grid_size):  # Gọi đệ quy để tiếp tục giải quyết
                            return True
                        board[row][col] = 0  # Quay lại nếu không tìm được giải pháp
                return False  # Nếu không thể điền vào ô này, trả về False
    return True  # Nếu không còn ô trống, tức là đã giải được
