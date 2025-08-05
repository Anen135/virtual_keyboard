import cv2
import HTM as htm

distance = {
    'draw': True,
    "line": {
        "color": (255, 0, 255),
        "bold": 2
    },
    "circle" : {
        "radius": 5,
        "color": (255, 0, 255),
        "bold": -1 # FILLED
    },
    "reference": {
        "ColorL": (255, 255, 255),
        "ColorC" : (0, 0, 255),
    }
}

class FingerCounterModule:
    def __init__(self, detector):
        self.detector = detector
        self.totalFingers = 0

    def drawFingerCount(self, img):
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(self.totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
    def fingersUp(self, img) -> list:
        Reference = self.detector.findDistance(5, 0, img, draw=distance['draw'], colorL=distance['reference']['ColorL'], colorC=distance['reference']['ColorC'])[0]
        rfl = Reference / 0.75
        return [
            self.detector.findDistance(4, 17, img)[0] > Reference / 1.3,
            self.detector.findDistance(8, 0, img)[0] > rfl,
            self.detector.findDistance(12, 0, img)[0] > rfl,
            self.detector.findDistance(16, 0, img)[0] > rfl,
            self.detector.findDistance(20, 0, img)[0] > rfl
        ]

def demo(app):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    print('Press "q" to quit')

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        app.detector.findHands(img)
        app.detector.findPosition(img)

        app.totalFingers = app.fingersUp(img).count(True) if app.detector.lmList else 0

        app.drawFingerCount(img)
        app.detector.drawFPS(img)

        cv2.imshow("IMG", img)
        if cv2.waitKey(1) & 0xFF in [27, 113, 233, 81]:  # esc, q, etc.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo(FingerCounterModule(htm.handDetector(detectionCon=0.75, maxHands=1)))
