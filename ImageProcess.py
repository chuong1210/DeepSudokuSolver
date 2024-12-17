import numpy as np
import cv2
from tensorflow.keras.models import load_model
import math
import copy

from sudokuSolverDiversity import solveDFS,is_valid_sudoku

def extrapolate_sudoku(image, model_name):
    '''Cho một hình ảnh Sudoku và mô hình mạng nơ-ron đã huấn luyện, trả về ma trận Sudoku được trích xuất từ hình ảnh'''
    # Import mô hình đã được huấn luyện
    model = load_model(model_name)

    # Đọc hình ảnh Sudoku từ file
 
    # Thay đổi kích thước hình ảnh để phù hợp hơn
    img_height = image.shape[0]
    img_width = image.shape[1]

    if img_height > img_width:
        image = cv2.resize(image, (800, 1000))
    elif img_height < img_width:
        image = cv2.resize(image, (1000, 800))
    else:
        image = cv2.resize(image, (800, 800))
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Làm mờ hình ảnh để giảm nhiễu và giúp nhận dạng số dễ hơn
    blur = cv2.GaussianBlur(gray, (13, 13), 0)

    # Áp dụng ngưỡng thích ứng để chuyển thành ảnh nhị phân
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

    # Làm xói mòn và giãn nở hình ảnh để tách các chữ số rõ ràng hơn
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(thresh, kernel)
    dilatation = cv2.dilate(erosion, kernel)

    # Đảo ngược màu sắc của hình ảnh (chuyển đen thành trắng và ngược lại)
    invert = cv2.bitwise_not(dilatation)

 # Tìm đường viền lớn nhất có dạng hình vuông (tương ứng với khung Sudoku)
    contours, hierarchy = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_rect_coord = []
    for c in contours_sorted:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        if len(approx) == 4:
            largest_rect_coord = approx
            #cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
            break

    # Trả về ma trận Sudoku rỗng nếu không tìm thấy đường viền nào
    if largest_rect_coord is None or len(largest_rect_coord) == 0:
        empty_sudoku = np.zeros((9, 9), np.int8)
        return empty_sudoku

    largest_rect_coord = largest_rect_coord.reshape(4,2)


    # Sắp xếp lại các tọa độ của đường viền
    sum_coord = largest_rect_coord.sum(1)
    diff_coord = np.diff(largest_rect_coord, axis=1)


    pt_A = largest_rect_coord[np.argmin(sum_coord)]  # Điểm trên cùng bên trái
    pt_B = largest_rect_coord[np.argmax(diff_coord)] # Điểm trên cùng bên phải
    pt_C = largest_rect_coord[np.argmax(sum_coord)]  # Điểm dưới cùng bên phải
    pt_D = largest_rect_coord[np.argmin(diff_coord)] # Điểm dưới cùng bên trái

    # Tính toán chiều dài của các cạnh lớn nhất
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    # Tạo ma trận các điểm đầu vào và đầu ra để biến đổi phối cảnh

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])

    # Biến đổi phối cảnh để có được hình vuông Sudoku
    transf = cv2.getPerspectiveTransform(input_pts, output_pts)
    warp = cv2.warpPerspective(invert, transf, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    grid = cv2.resize(warp, (252, 252))

    # Cắt hình ảnh thành các ô vuông (mỗi ô 28x28 pixel)
    cell = np.zeros([9, 9, 28, 28])

    for i in range(9):
        for j in range(9):
            cell[i][j] = grid[0+28*i:28+28*i, 0+28*j:28+28*j]

    # Tạo ma trận Sudoku để lưu các số nhận dạng được
    sudoku_grid = np.zeros((9, 9), np.int8)

    for i in range(9):
        for j in range(9):
            im = cell[i][j]

            # Chuyển hình ảnh sang dạng nhị phân để tìm đường viền số
            im = cv2.convertScaleAbs(im)

            # Tìm các thành phần kết nối trong ô
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im)

            digit_contour = []
            for k in range(0, labels.max() + 1):
                im_mask = cv2.compare(labels, k, cv2.CMP_EQ)

                cell_contours, hierarchy = cv2.findContours(im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in cell_contours:
                    for pt in c:
                        if np.sqrt((pt[0][0] - 14)**2 + (pt[0][1] - 14)**2) < 6: # change the values 6 and 8 if a number is not detected
                            if len(c) > 0 and len(c) <8:
                                continue
                            else:
                                digit_contour = c
                                break
            # check if there is a digit
            # Nếu không có đường viền số nào, gán ô là 0
            if len(digit_contour) == 0:
                sudoku_grid[i][j] = 0
                continue

                        
            # Tạo hình chữ nhật bao quanh số

            x, y, w, h = cv2.boundingRect(digit_contour)

            # Tạo mặt nạ đen, chỉ giữ lại phần chứa số
            mask = np.zeros_like(im)
            mask[y:y + h, x:x + w] = 1

            # Áp dụng mặt nạ để làm sạch hình ảnh
            clean_im = cv2.bitwise_and(im, im, mask=mask)

            # Cắt riêng phần chứa số
            digit_image = clean_im[y:y + h, x:x + w].copy()

            # before predicting the digit, check if there are at least 5 white pixels in the digit image, otherwise is considered noise
            # Kiểm tra nếu số lượng pixel trắng quá ít thì coi là nhiễu

            pixel_threshold = 5
            pixel_count = 0
            for m in range(digit_image.shape[0]):
                for n in range(digit_image.shape[1]):
                    if digit_image[m][n] == 255:
                        pixel_count += 1
            if pixel_count < pixel_threshold:
                sudoku_grid[i][j] = 0
                continue

            # Tạo ảnh giống định dạng MNIST (28x28)
            mnist_image = np.zeros((28, 28), float)  # Sử dụng float thay vì np.float

             # Đặt số vào giữa ảnh 28x28
            mean_row = int((28 - digit_image.shape[0]) / 2)
            mean_col = int((28 - digit_image.shape[1]) / 2)
            mnist_image[mean_row:mean_row + digit_image.shape[0], mean_col:mean_col + digit_image.shape[1]] = digit_image

            #  Định hình lại ảnh để phù hợp với mô hình (1, 28, 28, 1)
            mnist_image = np.expand_dims(mnist_image, axis=0)
            mnist_image = np.expand_dims(mnist_image, axis=3)

            # predict the digit using the model
            digit_prediction = model.predict(mnist_image)
            cell_value = np.argmax(digit_prediction, axis=1)

            # gán số dự đoán vào ô
            sudoku_grid[i][j] = cell_value


    return sudoku_grid, largest_rect_coord, transf, (maxWidth, maxHeight)

def resize_image(image, max_width=800, max_height=800):
    '''Hàm thu nhỏ ảnh về kích thước tối đa mà vẫn giữ tỉ lệ khung hình'''
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height))

def displayImageSolution(image, solution, original_grid,largest_rect_coord):
    # Đọc ảnh gốc

    

    # Thay đổi kích thước hình ảnh để phù hợp hơn
    img_height = image.shape[0]
    img_width = image.shape[1]

    if img_height > img_width:
        image = cv2.resize(image, (800, 1000))
    elif img_height < img_width:
        image = cv2.resize(image, (1000, 800))
    else:
        image = cv2.resize(image, (800, 800))
    # Trích xuất Sudoku grid và tọa độ
    
    if largest_rect_coord is None or len(largest_rect_coord) == 0:
        return image  # Trả về ảnh gốc nếu không tìm thấy lưới Sudoku
    
    # Sắp xếp tọa độ
    largest_rect_coord = largest_rect_coord.reshape(4,2)
    sum_coord = largest_rect_coord.sum(1)
    diff_coord = np.diff(largest_rect_coord, axis=1)
    
    pt_A = largest_rect_coord[np.argmin(sum_coord)]  # Điểm trên cùng bên trái
    pt_B = largest_rect_coord[np.argmax(diff_coord)] # Điểm trên cùng bên phải
    pt_C = largest_rect_coord[np.argmax(sum_coord)]  # Điểm dưới cùng bên phải
    pt_D = largest_rect_coord[np.argmin(diff_coord)] # Điểm dưới cùng bên trái
    
    # Tính kích thước của lưới Sudoku
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    
    # Vẽ đường viền xanh quanh lưới Sudoku
    cv2.polylines(image, [largest_rect_coord], True, (0, 255, 0), 3)
    
    # Tính toán kích thước của mỗi ô
    cell_width = maxWidth / 9
    cell_height = maxHeight / 9
    
    # Vẽ số lên ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(cell_width, cell_height) / 30.0  # Điều chỉnh kích thước font
    font_thickness = 2
    font_color = (0, 0, 255)  # Màu đỏ
    
    for i in range(9):
        for j in range(9):
            if original_grid[i][j] == 0:  # Chỉ vẽ số vào ô trống
                number = solution[i][j]
                if number != 0:
                    # Tính toán vị trí để vẽ số
                    x = int(pt_A[0] + j * cell_width + cell_width / 2)
                    y = int(pt_A[1] + i * cell_height + cell_height / 2)
                    
                    # Lấy kích thước của text
                    (text_width, text_height), _ = cv2.getTextSize(str(number), font, font_scale, font_thickness)
                    
                    # Điều chỉnh vị trí để text nằm giữa ô
                    text_x = int(x - text_width / 2)
                    text_y = int(y + text_height / 2)
                    
                    cv2.putText(image, str(number), (text_x, text_y), font, font_scale, font_color, font_thickness)
    
    return image





def extrapolate_sudoku_myself(image, model_path):
    # Convert frame to grayscale if it's not already
        '''Cho một hình ảnh Sudoku và mô hình mạng nơ-ron đã huấn luyện, trả về ma trận Sudoku được trích xuất từ hình ảnh'''
        # Import mô hình đã được huấn luyện
        model = load_model(model_path)

        # Đọc hình ảnh Sudoku từ file
    

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Làm mờ hình ảnh để giảm nhiễu và giúp nhận dạng số dễ hơn
        blur = cv2.GaussianBlur(gray, (13, 13), 0)

        # Áp dụng ngưỡng thích ứng để chuyển thành ảnh nhị phân
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

        # Làm xói mòn và giãn nở hình ảnh để tách các chữ số rõ ràng hơn
        kernel = np.ones((3,3), np.uint8)
        erosion = cv2.erode(thresh, kernel)
        dilatation = cv2.dilate(erosion, kernel)

        # Đảo ngược màu sắc của hình ảnh (chuyển đen thành trắng và ngược lại)
        invert = cv2.bitwise_not(dilatation)

    # Tìm đường viền lớn nhất có dạng hình vuông (tương ứng với khung Sudoku)
        contours, hierarchy = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

        largest_rect_coord = []
        for c in contours_sorted:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
            if len(approx) == 4:
                largest_rect_coord = approx
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
                break

        # Trả về ma trận Sudoku rỗng nếu không tìm thấy đường viền nào
        if largest_rect_coord is None or len(largest_rect_coord) == 0:
            empty_sudoku = np.zeros((9, 9), np.int8)
            return empty_sudoku

        largest_rect_coord = largest_rect_coord.reshape(4,2)


        # Sắp xếp lại các tọa độ của đường viền
        sum_coord = largest_rect_coord.sum(1)
        diff_coord = np.diff(largest_rect_coord, axis=1)


        pt_A = largest_rect_coord[np.argmin(sum_coord)]  # Điểm trên cùng bên trái
        pt_B = largest_rect_coord[np.argmax(diff_coord)] # Điểm trên cùng bên phải
        pt_C = largest_rect_coord[np.argmax(sum_coord)]  # Điểm dưới cùng bên phải
        pt_D = largest_rect_coord[np.argmin(diff_coord)] # Điểm dưới cùng bên trái

        # Tính toán chiều dài của các cạnh lớn nhất
        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        # Tạo ma trận các điểm đầu vào và đầu ra để biến đổi phối cảnh

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])

        # Biến đổi phối cảnh để có được hình vuông Sudoku
        transf = cv2.getPerspectiveTransform(input_pts, output_pts)
        warp = cv2.warpPerspective(invert, transf, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
        grid = cv2.resize(warp, (252, 252))

        # Cắt hình ảnh thành các ô vuông (mỗi ô 28x28 pixel)
        cell = np.zeros([9, 9, 28, 28])

        for i in range(9):
            for j in range(9):
                cell[i][j] = grid[0+28*i:28+28*i, 0+28*j:28+28*j]

        # Tạo ma trận Sudoku để lưu các số nhận dạng được
        sudoku_grid = np.zeros((9, 9), np.int8)

        for i in range(9):
            for j in range(9):
                im = cell[i][j]

                # Chuyển hình ảnh sang dạng nhị phân để tìm đường viền số
                im = cv2.convertScaleAbs(im)

                # Tìm các thành phần kết nối trong ô
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im)

                digit_contour = []
                for k in range(0, labels.max() + 1):
                    im_mask = cv2.compare(labels, k, cv2.CMP_EQ)

                    cell_contours, hierarchy = cv2.findContours(im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for c in cell_contours:
                        for pt in c:
                            if np.sqrt((pt[0][0] - 14)**2 + (pt[0][1] - 14)**2) < 6: # change the values 6 and 8 if a number is not detected
                                if len(c) > 0 and len(c) <8:
                                    continue
                                else:
                                    digit_contour = c
                                    break
                # check if there is a digit
                # Nếu không có đường viền số nào, gán ô là 0
                if len(digit_contour) == 0:
                    sudoku_grid[i][j] = 0
                    continue

                            
                # Tạo hình chữ nhật bao quanh số

                x, y, w, h = cv2.boundingRect(digit_contour)

                # Tạo mặt nạ đen, chỉ giữ lại phần chứa số
                mask = np.zeros_like(im)
                mask[y:y + h, x:x + w] = 1

                # Áp dụng mặt nạ để làm sạch hình ảnh
                clean_im = cv2.bitwise_and(im, im, mask=mask)

                # Cắt riêng phần chứa số
                digit_image = clean_im[y:y + h, x:x + w].copy()

                # before predicting the digit, check if there are at least 5 white pixels in the digit image, otherwise is considered noise
                # Kiểm tra nếu số lượng pixel trắng quá ít thì coi là nhiễu

                pixel_threshold = 5
                pixel_count = 0
                for m in range(digit_image.shape[0]):
                    for n in range(digit_image.shape[1]):
                        if digit_image[m][n] == 255:
                            pixel_count += 1
                if pixel_count < pixel_threshold:
                    sudoku_grid[i][j] = 0
                    continue

                # Tạo ảnh giống định dạng MNIST (28x28)
                mnist_image = np.zeros((28, 28), float)  # Sử dụng float thay vì np.float

                # Đặt số vào giữa ảnh 28x28
                mean_row = int((28 - digit_image.shape[0]) / 2)
                mean_col = int((28 - digit_image.shape[1]) / 2)
                mnist_image[mean_row:mean_row + digit_image.shape[0], mean_col:mean_col + digit_image.shape[1]] = digit_image

                #  Định hình lại ảnh để phù hợp với mô hình (1, 28, 28, 1)
                mnist_image = np.expand_dims(mnist_image, axis=0)
                mnist_image = np.expand_dims(mnist_image, axis=3)

                # predict the digit using the model
                digit_prediction = model.predict(mnist_image)
                cell_value = np.argmax(digit_prediction, axis=1)

                # gán số dự đoán vào ô
                sudoku_grid[i][j] = cell_value


        return sudoku_grid, largest_rect_coord



def is_sudoku_present(frame):
    '''Kiểm tra xem frame có chứa lưới Sudoku hay không.'''
    # Chuyển ảnh về grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

    # Tìm đường viền trong ảnh đã xử lý
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Nếu tìm thấy đường viền có 4 cạnh (hình chữ nhật hoặc hình vuông)
        if len(approx) == 4:
            # Sắp xếp lại các điểm để xác định chiều dài và chiều rộng của hình chữ nhật
            approx = approx.reshape(4, 2)
            rect_width = np.linalg.norm(approx[0] - approx[1])
            rect_height = np.linalg.norm(approx[1] - approx[2])

            # Kiểm tra tỷ lệ giữa chiều rộng và chiều cao (hình vuông hoặc hình chữ nhật gần vuông)
            aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)


            # Lưới Sudoku thường có tỷ lệ gần 1 (gần vuông)
            if 0.9 < aspect_ratio < 1.1:
                # Kiểm tra diện tích của hình chữ nhật
                area = cv2.contourArea(contour)

                # Kiểm tra xem diện tích có trong phạm vi hợp lý cho lưới Sudoku không
                if 10000 < area < 500000:  # Điều chỉnh các giá trị này tùy theo kích thước của lưới Sudoku trong video
                    if area > largest_area:
                        largest_area = area
                        largest_contour = approx

        # Nếu tìm thấy hình vuông lớn nhất, tô màu đỏ vào vùng đó
                    if largest_contour is not None:
             


                    

                        return True  # Đã phát hiện lưới Sudoku
   
    return False  # Không phát hiện lưới Sudoku




def display_sudoku_on_frame(frame, model_path):

    sudoku_grid, largest_rect_coord = extrapolate_sudoku_myself(frame, model_path)
    print(sudoku_grid)
    if(is_valid_sudoku(sudoku_grid)):



        # Xác định các điểm góc của lưới Sudoku
        largest_rect_coord = largest_rect_coord.reshape(4, 2)

        sum_coord = largest_rect_coord.sum(1)
        diff_coord = np.diff(largest_rect_coord, axis=1)

        pt_A = largest_rect_coord[np.argmin(sum_coord)]
        pt_B = largest_rect_coord[np.argmax(diff_coord)]
        pt_C = largest_rect_coord[np.argmax(sum_coord)]
        pt_D = largest_rect_coord[np.argmin(diff_coord)]

        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])

        # Tính ma trận biến đổi phối cảnh và ma trận nghịch đảo
        transf = cv2.getPerspectiveTransform(input_pts, output_pts)
        inv_transf = cv2.getPerspectiveTransform(output_pts, input_pts)

        grid_cell_height = maxHeight // 9
        grid_cell_width = maxWidth // 9

        # Vẽ kết quả lên frame gốc
        for i in range(9):
            for j in range(9):
                num = sudoku_grid[i][j]
                if num != 0:
                    # Tính tọa độ trung tâm của mỗi ô trong warp
                    x_warp = int(j * grid_cell_width + grid_cell_width / 2)
                    y_warp = int(i * grid_cell_height + grid_cell_height / 2)

                    # Chuyển đổi tọa độ từ warp về frame gốc
                    point_warp = np.array([[[x_warp, y_warp]]], dtype=np.float32)
                    point_frame = cv2.perspectiveTransform(point_warp, inv_transf)[0][0]

                    # Vẽ số lên frame gốc
                    cv2.putText(frame, str(num), (int(point_frame[0]), int(point_frame[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return largest_rect_coord, frame, sudoku_grid,inv_transf, maxWidth, maxHeight


    return None, frame, None, None, 0, 0



def display_sudoku_on_frame1(frame, model_path):


        sudoku_grid, largest_rect_coord = extrapolate_sudoku_myself(frame, model_path)
        solve_grid = copy.deepcopy(sudoku_grid)

        solveDFS(solve_grid)
        

        # Vẽ lại các số lên frame
        largest_rect_coord = largest_rect_coord.reshape(4, 2)

        sum_coord = largest_rect_coord.sum(1)
        diff_coord = np.diff(largest_rect_coord, axis=1)

        pt_A = largest_rect_coord[np.argmin(sum_coord)]
        pt_B = largest_rect_coord[np.argmax(diff_coord)]
        pt_C = largest_rect_coord[np.argmax(sum_coord)]
        pt_D = largest_rect_coord[np.argmin(diff_coord)]

        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                 [0, maxHeight - 1],
                                 [maxWidth - 1, maxHeight - 1],
                                 [maxWidth - 1, 0]])

        transf = cv2.getPerspectiveTransform(input_pts, output_pts)
        warp = cv2.warpPerspective(frame, transf, (maxWidth, maxHeight))

        grid_cell_height = maxHeight // 9
        grid_cell_width = maxWidth // 9

        # Vẽ kết quả lên hình ảnh Sudoku
        for i in range(9):
            for j in range(9):
                x = int(j * grid_cell_width)  # Chắc chắn là số nguyên
                y = int(i * grid_cell_height)  # Chắc chắn là số nguyên
                num = solve_grid[i][j]

                if num != 0:
                    cv2.putText(warp, str(num), (x + int(grid_cell_width // 4), y + int(grid_cell_height // 1.5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # Vẽ kết quả lên frame gốc
        cv2.imshow("Sudoku Solution", warp)
        return largest_rect_coord,frame,sudoku_grid

      


