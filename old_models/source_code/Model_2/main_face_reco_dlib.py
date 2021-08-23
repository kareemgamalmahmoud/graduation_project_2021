import cv2
import dlib
from scipy.spatial import distance
import winsound

def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# type of the font
font = cv2.FONT_HERSHEY_SIMPLEX

while 1:

    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n1,n2 in zip(range(36,42),range(42,48)):
            x_n1 = face_landmarks.part(n1).x
            y_n1 = face_landmarks.part(n1).y
            leftEye.append((x_n1,y_n1))
            next_point = n1+1
            if n1 == 41:
                next_point = 36
            x2_n1 = face_landmarks.part(next_point).x
            y2_n1 = face_landmarks.part(next_point).y
            cv2.line(frame,(x_n1,y_n1),(x2_n1,y2_n1),(255,255,255),1)

            # -------------------------------------------

            x_n2 = face_landmarks.part(n2).x
            y_n2 = face_landmarks.part(n2).y
            rightEye.append((x_n2, y_n2))
            next_point = n2 + 1
            if n2 == 47:
                next_point = 42
            x2_n2 = face_landmarks.part(next_point).x
            y2_n2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x_n2, y_n2), (x2_n2, y2_n2), (255, 255, 255), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        if EAR<0.26:
            cv2.putText(frame, "closed", (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            winsound.Beep(1000, 100)
        else:
            cv2.putText(frame, "open", (50, 50), font, 3, (0, 255, 0), 2, cv2.LINE_4)

        print("the ratio : ",EAR)
        print(" ")

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()