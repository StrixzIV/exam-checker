import cv2
import numpy as np

image = cv2.imread('./assets/neet-omr-sheet.webp')

roi = image[120:1315, 560:700]
grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
inp = cv2.GaussianBlur(grey, ksize = (15, 15), sigmaX = 15)

(thres, res) = cv2.threshold(inp, 110, 255, cv2.THRESH_BINARY)

res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype = np.uint8), iterations = 2)
res = cv2.dilate(res, kernel = (3, 3))

(contours, hierarchy) = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
drawed_contours = cv2.drawContours(roi, contours[1:], -1, (0, 255, 0), 2)

print(f'Found: {len(contours[1:])} of 35')

cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()