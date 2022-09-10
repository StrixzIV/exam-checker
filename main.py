import cv2
import numpy as np

from utils import id_block_read, read_answer, find_paper

imlist = ['./assets/ans_sheet1.jpg', './assets/ans_sheet2.jpg']

for img in imlist:

    image = cv2.imread(img)

    answer_sheet = cv2.resize(find_paper(image), (827, 1669))
    student_id = id_block_read(answer_sheet)

    print(f'ID: {student_id}')
    print(f'Answer: \n{read_answer(answer_sheet[690:845, 105:190], 5, debug = False)}')

cv2.waitKey(0)
cv2.destroyAllWindows()