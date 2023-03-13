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
        
        readed.append((question_idx + 1, choice_idx + 1))
    
    read = [None] * n_questions
    
    for (n, choice) in readed:
        read[n - 1] = choice
    
    return read


def ans_block_read(image: np.ndarray, n_block: int) -> list[int]:
    
    '''
        Read answer from \'n\' blocks of the main answer sheet.
    '''
    
    answers = []

    if n_block <= 5:
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 105:190]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
    
    elif n_block > 5 and n_block <= 9:

        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 105:190]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))

        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 245:330]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break
            
            answers.append(read)
    
    elif n_block > 9 and n_block <= 13:
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 105:190]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))

        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 245:330]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 385:470]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
    
    elif n_block > 13 and n_block <= 17:
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 105:190]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))

        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 245:330]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 385:470]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 385:470]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
    
    elif n_block > 17 and n_block <= 21:
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 105:190]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))

        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 245:330]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 385:470]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 525:610]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
        
        for i in range(0, n_block - 1):

            img = image[690 + (i * 190):845 + (i * 190), 665:750]
            read = read_answer(img, 5, debug = False)

            if set(read) == {None}:
                break

            answers.append(read_answer(img, 5, debug = False))
    
    elif n_block > 21:
        raise ValueError("n_block must be less than or equal to 20 blocks")

    return [j for i in answers for j in i]
    
    
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