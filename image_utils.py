import cv2
import numpy as np

from uuid import uuid4

def biggest_contour(contours: list) -> np.ndarray:

    for contour in contours:

        if cv2.contourArea(contour) > 1000:

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx
                
    return None


def find_paper(image: np.ndarray) -> np.ndarray:
    
    '''
        Find an answer sheet in the image and auto crop it.
    '''
    # define read answersheet image output size
    (max_width, max_height) = (827, 1669)
    
    img_original = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Standard blur to smooth out noise but retain document edges
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Stronger Canny thresholds to ignore paper texture and focus on strong edges
    edged = cv2.Canny(gray, 75, 200)

    # 3. Dilate and erode to close any small gaps in the paper's outline
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)

    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    biggest = biggest_contour(contours)

    # Safety catch in case no 4-point contour is found
    if biggest is None:
        print("Warning: Could not find paper outline. Returning original image.")
        return cv2.resize(img_original, (max_width, max_height))

    cv2.drawContours(image, [biggest], -1, (0, 255, 0), 3)

    # Pixel values in the original image
    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))
    
    return img_output


def read_answer(roi: np.ndarray, n_questions: int, debug: bool = True) -> list[int]:
    '''
        Read answer mark from a specific region of the answer sheet and return a result as a list.
    '''
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    inp = cv2.GaussianBlur(grey, ksize=(3, 3), sigmaX=1)

    (_, res) = cv2.threshold(inp, 150, 255, cv2.THRESH_BINARY_INV)

    # Dialed down to a 3x3 kernel so lightly shaded pencil marks don't get accidentally erased
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    res = cv2.dilate(res, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)
    
    if debug:
        cv2.imshow('roi_debug', res)
        cv2.waitKey(0)

    (contours, _) = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    readed = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
            
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # Calculate the exact center of the bubble instead of the top-left edge
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        
        if debug:
            print(f"Mark center at cx:{cx}, cy:{cy}")
        
        # 147px total height / 5 questions = 29.4px per row
        question_idx = int(cy / 29.4)
        
        # 85px total width / 4 choices = 21.25px per column
        choice_idx = int(cx / 21.25)

        if question_idx < 0 or question_idx >= n_questions:
            continue
        
        if choice_idx < 0 or choice_idx > 3:
            continue
        
        readed.append((question_idx + 1, choice_idx + 1))
    
    read = [None] * n_questions
    
    for (n, choice) in readed:
        read[n - 1] = choice
    
    return read


def ans_block_read(image: np.ndarray, n_questions: int = 100) -> list[int]:
    
    '''
        Read answers from all blocks of the main answer sheet.
    '''
    if n_questions > 100 or n_questions < 1:
        raise ValueError("n_questions must be between 1 and 100.")

    # Shifted X-ranges strictly RIGHT (starting at 120 instead of 115 or 110)
    # This safely dodges the printed numbers on the left for both scans and photos.
    col_x_ranges = [
        (120, 205),   # Column 1
        (260, 345),   # Column 2
        (400, 485),   # Column 3
        (540, 625),   # Column 4
        (680, 765),   # Column 5
    ]

    row_y_ranges = [
        ( 698,  845),  # Row-group 1
        ( 888, 1035),  # Row-group 2
        (1078, 1225),  # Row-group 3
        (1268, 1415),  # Row-group 4
    ]

    answers = []

    for (x_start, x_end) in col_x_ranges:
        for (y_start, y_end) in row_y_ranges:

            if len(answers) >= n_questions:
                break

            roi = image[y_start:y_end, x_start:x_end]
            
            # This calls your updated `read_answer` with the cx/cy logic
            block_answers = read_answer(roi, 5, debug=False)

            if set(block_answers) == {None}:
                answers.extend([None] * 5)
            else:
                answers.extend(block_answers)

        if len(answers) >= n_questions:
            break

    return answers[:n_questions]

    
def id_block_read(image: np.ndarray, debug: bool = True) -> int:

    '''
        Read the ID from the id section of the answer sheet image
    '''
    
    img = image[350:625, 305:375]
        
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inp = cv2.GaussianBlur(grey, ksize = (3, 3), sigmaX = 1)

    (_, res) = cv2.threshold(inp, 150, 255, cv2.THRESH_BINARY_INV)

    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
    res = cv2.dilate(res, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)

    id_str = ''
    
    col_width = 23

    for i in range(3):
        
        col_img = res[:, i * col_width : (i + 1) * col_width]
        
        # RETR_EXTERNAL only grabs the outer shape of the blobs
        (contours, _) = cv2.findContours(col_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if debug:
            from uuid import uuid4
            cv2.imshow(str(uuid4()), col_img)
            cv2.waitKey(0)
        
        # Filter out any tiny leftover noise specks by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20]
        
        if not valid_contours:
            continue
            
        # Grab the largest blob in this column (the filled bubble)
        largest_cnt = max(valid_contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_cnt)
        
        if debug:
            print(f"Col {i} Y-coord: {y}")
        
        if y in range(0, 26):
            id_str += '1'
            
        elif y in range(26, 51):
            id_str += '2'
        
        elif y in range(51, 76):
            id_str += '3'
        
        elif y in range(76, 101):
            id_str += '4'
        
        elif y in range(101, 126):
            id_str += '5'
        
        elif y in range(126, 151):
            id_str += '6'
        
        elif y in range(151, 176):
            id_str += '7'
        
        elif y in range(176, 201):
            id_str += '8'
        
        elif y in range(201, 226):
            id_str += '9'
        
        elif y in range(226, 285):
            id_str += '0'
    
    if id_str == '':
        return 0
        
    return int(id_str)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    
    '''
        Rotate image for n degree.
    '''
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result