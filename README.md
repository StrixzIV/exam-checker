# exam-checker

A Final project for semester 1 of E-AI SPSM. 

**Exam checker** an answer sheet marker build with **[OpenCV](https://github.com/opencv/opencv-python)**.

## Installation

clone the project with git 

```sh
git clone https://github.com/strixziv/exam-checker.git
```

## Dependentcies & tools

This project required Python 3.9 to function correctly.

To install the dependentcies, run the following command:

```sh
cd /path/to/exam-checker
pip install --upgrade -r requirements.txt
```

## How to use

*Please make sure you have already install all of the required dependcies first. If not please follow **[this](#Dependentcies-&-tools)** instructions.

1. Take an image of the answer sheet you want to mark
2. Put the images in **[./assets/images/](./assets/images/)**
3. Run the program by running `python main.py`

The output excel spread sheet will be stored at the project's root directory.