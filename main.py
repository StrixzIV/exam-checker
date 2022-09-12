import os
import cv2

from utils import id_block_read, read_answer, find_paper, ans_block_read

imlist = ['./assets/images/' + path for path in os.listdir('./assets/images')]

print('*' * 100)

for img in imlist:

    image = cv2.imread(img)

    answer_sheet = cv2.resize(find_paper(image), (827, 1669))
    student_id = id_block_read(answer_sheet, debug = False)
    answers = ans_block_read(answer_sheet, 5)

    print(f'ID: {student_id}, {img}')
    print(answers, len(answers))

    print('*' * 100)

# cv2.waitKey(0)
cv2.destroyAllWindows()