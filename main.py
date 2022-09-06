import cv2
import numpy as np

image = cv2.imread('./assets/neet-omr-sheet.webp')

def read_answer(roi: any, n_questions: int) -> list[int]:
    
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
        

readed = read_answer(image[120:1315, 560:700], 35)
# readed2 = read_answer(image[120:1315, 560 + 200:700 + 200])
    
for n, choice in enumerate(readed, start = 1):
    print(f'{n}: {choice}')

cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()