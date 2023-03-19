## Face Detection with OpenCV


# Import OpenCV
import cv2
import time

# Loading the cascades
face_cascade = cv2.CascadeClassifier(
    '/Users/prashantkondayapalepu/PycharmProjects/pythonProject1/frontalface.xml')
eye_cascade = cv2.CascadeClassifier(
    '/Users/prashantkondayapalepu/PycharmProjects/pythonProject1/eye.xml')
smile_cascade = cv2.CascadeClassifier(
    '/Users/prashantkondayapalepu/PycharmProjects/pythonProject1/smile.xml')

# Face Detector
def face_detector(grayScale, originalFrame):
    faces = face_cascade.detectMultiScale(grayScale, 1.3, 5)  # Coordinates of detected faces
    for (x1, y1, w1, h1) in faces:  # For loop to iterate over detected faces
        cv2.rectangle(originalFrame,
                      (x1, y1),  # Upper Left Corner
                      (x1 + w1, y1 + h1),  # Lower Right Corner
                      (255, 255, 0), 3)  # Drawing rectangle over face on Original Frame
        roi_gray = grayScale[y1:y1 + h1, x1:x1 + w1]
        roi_color = originalFrame[y1:y1 + h1, x1:x1 + w1]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 21)  # Coordinates of detected eyes
        for (x2, y2, w2, h2) in eyes:  # For loop to iterate over detected eyes
            cv2.rectangle(roi_color,
                          (x2, y2),  # Upper Left Corner
                          (x2 + w2, y2 + h2),  # Lower Right Corner
                          (255, 0, 0), 2)  # Drawing rectangle over eye on Original Frame

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)  # Coordinates of detected smile
        count =0
        for (sx, sy, sw, sh) in smiles:  # For loop to iterate over detected smile
            count+=1
            cv2.rectangle(roi_color,
                          (sx, sy),  # Upper Left Corner
                          (sx + sw, sy + sh),  # Lower Right Corner
                          (250, 0, 0), 2)  # Drawing rectangle over eye on Original Frame
            if count!= 1:
             font = cv2.FONT_HERSHEY_SIMPLEX

            # org
             org = (50, 50)

            # fontScale
             fontScale = 1

            # Blue color in BGR
             color = (255, 0, 0)

            # Line thickness of 2 px
             thickness = 2
             org = (int(sx+100), int(sy))

             cv2.putText(roi_color, 'Happy :)', org, font, fontScale, color, thickness)
            if sh<60:
                font = cv2.FONT_HERSHEY_SIMPLEX

                # org
                org = (50, 50)

                # fontScale
                fontScale = 1

                # Blue color in BGR
                color = (0, 0, 255)

                # Line thickness of 2 px
                thickness = 2
                org = (int(sx ), int(sy-40))

                cv2.putText(roi_color, 'Unhappy :(', org ,font, fontScale, color,thickness)
                cv2.rectangle(roi_color,
                 (sx, sy),  # Upper Left Corner
                 (sx + sw, sy + sh),  # Lower Right Corner
                    (0, 0, 250), 2)  # Drawing rectangle over eye on Or
        return originalFrame


# Accessing webcam to detect face
video_capture = cv2.VideoCapture(0)
print('Detecting camera...')
time.sleep(20)

while (video_capture.isOpened()):
    re, frame = video_capture.read()

    if re:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = face_detector(gray, frame)
        if canvas is None:continue
        cv2.imshow('Video', canvas)
    else:
        print('No camera detected.')

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()