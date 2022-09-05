import cv2
import numpy as np

image = cv2.imread('./assets/neet-omr-sheet.webp')

img = cv2.resize(image, (700, 700))

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()