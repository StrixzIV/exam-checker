import cv2
import numpy as np

from utils import id_block_read, read_answer, find_paper, ans_block_read

imlist = ['./assets/ans_sheet1.jpg', './assets/ans_sheet2.jpg']

print('*' * 100)

for img in imlist:

    image = cv2.imread(img)

    answer_sheet = cv2.resize(find_paper(image), (827, 1669))
    student_id = id_block_read(answer_sheet)
    answers = ans_block_read(answer_sheet, 4)

    print(f'ID: {student_id}')
    print(answers)

    print('*' * 100)

cv2.waitKey(0)
cv2.destroyAllWindows()