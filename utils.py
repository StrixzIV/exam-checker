import cv2
import numpy as np

def biggest_contour(contours: any) -> any:
    
    biggest = np.array([])
    max_area = 0
    
    for i in contours:
        
        area = cv2.contourArea(i)
        
        if area > 1000:
            
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
                
    return biggest


def find_paper(image: np.ndarray) -> np.ndarray:
    
    '''
        Find an answer sheet in the image and auto cropped
    '''
    
    img_original = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 10, 20)

    (contours, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    (top_left, top_right, bottom_right, bottom_left) = input_points

    # Output image size
    max_width = 827
    # max_height = max(int(right_height), int(left_height))
    max_height = 1669

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))
    
    return img_output


def read_answer(roi: np.ndarray, n_questions: int) -> list[int]:
    
    '''
        Read answer mark from a specific region of the answer sheet and return a result as a list.
    '''
    
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    inp = cv2.GaussianBlur(grey, ksize = (15, 15), sigmaX = 25)

    (_, res) = cv2.threshold(inp, 110, 255, cv2.THRESH_BINARY)

    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype = np.uint8), iterations = 2)
    res = cv2.dilate(res, kernel = (3, 3))

    (contours, hierarchy) = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(roi, contours[1:n_questions], -1, (0, 255, 0), 2)

    readed = []

    for cnt in contours[1:][::-1]:
        
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        if x in range(0, 15):
            readed.append((int(y // 33.5) + 1, 1))
            
        elif x in range(35, 50):
            readed.append((int(y // 33.5) + 1, 2))
            
        elif x in range(70, 85):
            readed.append((int(y // 33.5) + 1, 3))
            
        elif x in range(105, 120):
            readed.append((int(y // 33.5) + 1, 4))
            
    read = [None] * n_questions
    
    for n, choice in readed:
        read[n - 1] = choice
    
    return read


def ans_block_read(image: np.ndarray, n_block: int) -> list:
    
    if n_block <= 4:
        
        for i in range(n_block):
            img = image[:, :n_block]
    
    elif n_block > 4 and n_block <= 8:
        pass
    
    elif n_block > 8 and n_block <= 12:
        pass
    
    elif n_block > 12 and n_block <= 16:
        pass
    
    elif n_block > 16 and n_block <= 20:
        pass
    
    elif n_block > 20:
        raise ValueError("n_block must be less than or equal to 20 blocks")
    
    
def id_block_read(image: np.ndarray) -> str:
    
    '''
        Read the ID from the id section of the answer sheet image
    '''
    
    img = image[340:620, 300:370]
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inp = cv2.GaussianBlur(grey, ksize = (5, 5), sigmaX = 1)

    res = cv2.adaptiveThreshold(inp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1)

    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype = np.uint8), iterations = 2)
    res = cv2.dilate(res, kernel = (3, 3))

    id_str = ''

    for i in range(1, 4):
        
        (contours, hierarchy) = cv2.findContours(res[:, :i * 20], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for cnt in (contours[1:][::-1]):
            
            if len(id_str) == 3:
                break
            
            (x, y, w, h) = cv2.boundingRect(cnt)
            
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
    
    return id_str


def rotate_image(image: np.ndarray, angle: int) -> any:
    
    '''
        Rotate image for n degree.
    '''
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result