import os
import cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from rich import print
from image_utils import id_block_read, find_paper, ans_block_read

imlist = ['./assets/images/' + path for path in os.listdir('./assets/images')]

correct_ans = []
datasets = []

print('Reading answers from the sheet...')

for img in tqdm(imlist, unit = 'Sheet'):

    image = cv2.imread(img)

    answer_sheet = cv2.resize(find_paper(image), (827, 1669))
    student_id = id_block_read(answer_sheet, debug = False)
    answers = ans_block_read(answer_sheet, 5)
    
    if student_id == 0:
        correct_ans = answers
    
    data = {'id': student_id, 'answers': answers}
    datasets.append(data.copy())

datasets = sorted(datasets, key = lambda data: data['id'])[1:]

print(f'Correct answers(ID = 0): {correct_ans}')
print(datasets)
# cv2.waitKey(0)
cv2.destroyAllWindows()

for (idx, data) in enumerate(datasets):
    
    datasets[idx]['answers_check'] = []
    
    for (base, student) in zip(correct_ans, data['answers']):
        datasets[idx]['answers_check'].append(base == student)
        
for data in datasets:
    (_, count) = np.unique(data['answers_check'], return_counts = True)
    data['correct'] = count[1]
    data['incorrect'] = count[0]
    
df = pd.DataFrame(datasets)
df['pass'] = ["Pass" if d >= 6 else "Not pass" for d in df['correct']]

df.to_excel('out.xlsx', index = False)