import cv2 as cv

cap = cv.VideoCapture(0)

faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

while (True):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (30, 30)
    )

    for (x, y, w, h) in face:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 5)

    cv.imshow("face", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()