import cv2
import numpy as np

from uuid import uuid4

def biggest_contour(contours: np.ndarray) -> np.ndarray:
    
    contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.015 * peri, True)
        
        if len(approx) == 4:
            return approx
    
    return None


def find_paper(image: np.ndarray) -> np.ndarray:
    
    '''
        Find an answer sheet in the image and auto cropped
    '''
    
    # define readed answersheet image output size
    (max_width, max_height) = (827, 1669)
    
    img_original = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 10, 20)

    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    biggest = biggest_contour(contours)

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
    inp = cv2.GaussianBlur(grey, ksize = (15, 15), sigmaX = 1)

    (_, res) = cv2.threshold(inp, 185, 255, cv2.THRESH_BINARY)

    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype = np.uint8), iterations = 3)
    res = cv2.dilate(res, kernel = (3, 3))
    
    if debug:
        cv2.imshow(str(uuid4()), res)
        cv2.waitKey(0)

    (contours, _) = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    readed = []

    for cnt in contours[::-1]:
        
        (x, y, _w, _h) = cv2.boundingRect(cnt)
        
        if debug:
            print(x, y)
        
        question_idx = int(y // 27)
        choice_idx = (x - 1) // 20

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

        The sheet is laid out as a grid:
          - 5 columns  (each covering 20 questions)
          - 4 row-groups per column (each covering 5 questions)
        Total: 5 × 4 × 5 = 100 questions

        Column x-ranges (left edge → right edge in the warped 827×1669 image):
          Col 1 (Q  1-20 ): x 105:190
          Col 2 (Q 21-40 ): x 245:330
          Col 3 (Q 41-60 ): x 385:470
          Col 4 (Q 61-80 ): x 525:610
          Col 5 (Q 81-100): x 665:750

        Row-group y-ranges (top → bottom):
          Group 1 (rows  1- 5 per col): y  690:845
          Group 2 (rows  6-10 per col): y  880:1035
          Group 3 (rows 11-15 per col): y 1070:1225
          Group 4 (rows 16-20 per col): y 1260:1415

        n_questions: how many questions to read (default 100).
                     Must be a multiple of 5 and <= 100.
    '''

    if n_questions > 100 or n_questions < 1:
        raise ValueError("n_questions must be between 1 and 100.")

    # X pixel ranges for each of the 5 answer columns
    col_x_ranges = [
        (105, 190),   # Column 1: Q  1–20
        (245, 330),   # Column 2: Q 21–40
        (385, 470),   # Column 3: Q 41–60
        (525, 610),   # Column 4: Q 61–80
        (665, 750),   # Column 5: Q 81–100
    ]

    # Y pixel ranges for each of the 4 row-groups inside every column.
    # We trim 8px off the top of each range to avoid capturing the horizontal
    # separator line that sits at the very top of each group block — that line
    # was being detected as a contour at y≈0, x≈0, corrupting question 1 in
    # every block with a spurious choice of 0.
    row_y_ranges = [
        ( 698,  845),  # Row-group 1: questions  1– 5 within each column
        ( 888, 1035),  # Row-group 2: questions  6–10 within each column
        (1078, 1225),  # Row-group 3: questions 11–15 within each column
        (1268, 1415),  # Row-group 4: questions 16–20 within each column
    ]

    answers = []

    for (x_start, x_end) in col_x_ranges:
        for (y_start, y_end) in row_y_ranges:

            if len(answers) >= n_questions:
                break

            roi = image[y_start:y_end, x_start:x_end]
            block_answers = read_answer(roi, 5, debug=False)

            # Stop early if the entire block is blank (no marks at all)
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
    
    img = image[340:625, 300:370]
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inp = cv2.GaussianBlur(grey, ksize = (3, 3), sigmaX = 1)

    (_, res) = cv2.threshold(inp, 178, 255, cv2.THRESH_BINARY)

    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype = np.uint8), iterations = 4)
    res = cv2.dilate(res, kernel = (5, 5), iterations = 3)

    id_str = ''

    for i in range(1, 4):
        
        (contours, _) = cv2.findContours(res[:, (i - 1) * 21:i * 21], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if debug:
            cv2.imshow(str(uuid4()), res[:, (i - 1) * 21:i * 21])
            cv2.waitKey(0)
        
        for cnt in (contours[1:][::-1]):
            
            if len(id_str) == 3:
                break
            
            (x, y, w, h) = cv2.boundingRect(cnt)
            
            if debug:
                print(y)
            
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
                
            elif y in range(226, 261):
                id_str += '0'
    
    return int(id_str)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    
    '''
        Rotate image for n degree.
    '''
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result