import cv2                                   # OpenCV, для работы с видео.
from HTM import handDetector  # Импортируем модуль для отслеживания рук.
import numpy as np                           # NumPy, для работы с массивами.    
import cvzone                                # cvzone, для удобной работы с OpenCV.
from pynput.keyboard import Controller, Key  # Импортируем контроллер клавиатуры для эмуляции нажатий клавиш.

cap = cv2.VideoCapture(0) # Инициализируем видеозахват с камеры.
cap.set(3, 1280)          # Устанавливаем ширину кадра.
cap.set(4, 720)           # Устанавливаем высоту кадра.

detector = handDetector(detectionCon=0.8, maxHands=1)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
        ["<", " ", "^"]]
finalText = ""
keyboard = Controller()

def drawALL(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(imgNew, (x, y, w, h), 20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
    return out

class Button():
    def __init__(self, pos, text, size=None):
        self.pos = pos
        self.size = size or [85, 85]
        self.text = text

buttonList = [
    Button([100 * j + 50, 100 * i + 50], key)
    for i, row in enumerate(keys)
    for j, key in enumerate(row)
]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    detector.findHands(img)
    if bboxInfo := detector.findPosition(img): detector.drawRectangle(img, bboxInfo)
    img = drawALL(img, buttonList)

    if detector.lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < detector.lmList[8][1] < x + w and y < detector.lmList[8][2] < y + h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                l, _ =  detector.findDistance(8, 0, img, draw=True)

                ## when clicked
                if l < 120:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    if button.text == "<":
                        finalText = finalText[:-1]
                        keyboard.press('\010')
                    elif button.text == "^":
                        finalText = ""
                        keyboard.press(Key.enter)
                        keyboard.release(Key.enter)
                    else:
                        finalText += button.text
                        keyboard.press(button.text)

    cv2.rectangle(img, (50, 710), (700, 610), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 690), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
    cv2.imshow("Keyboard", img)
    cv2.waitKey(1)