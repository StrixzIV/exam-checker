import os
import cv2

from tqdm import tqdm
from rich import print
from utils import id_block_read, read_answer, find_paper, ans_block_read

imlist = ['./assets/images/' + path for path in os.listdir('./assets/images')]

correct_ans = []
datasets = []

for img in imlist:

    image = cv2.imread(img)

    answer_sheet = cv2.resize(find_paper(image), (827, 1669))
    student_id = id_block_read(answer_sheet, debug = False)
    answers = ans_block_read(answer_sheet, 5)
    
    if student_id == 0:
        correct_ans = answers
    
    data = {'id': student_id, 'answers': answers}
    datasets.append(data.copy())

datasets = sorted(datasets, key = lambda data: data['id'])

print(f'Correct answers(ID = 0): {correct_ans}')
print(datasets)
# cv2.waitKey(0)
cv2.destroyAllWindows()