import cv2
import numpy as np

from utils import id_block_read, read_answer, find_paper

image = cv2.imread('./assets/ans_sheet1.jpg')

answer_sheet = cv2.resize(find_paper(image), (827, 1669))
student_id = id_block_read(answer_sheet)

print(student_id)

cv2.imshow('image', answer_sheet)

cv2.waitKey(0)
cv2.destroyAllWindows()