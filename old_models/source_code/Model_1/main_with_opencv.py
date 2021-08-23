from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import winsound #this works only for windows

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(148, 148, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])




history = models.load_model('close_and_open_small_1.h5')

history.summary()


imag_array = cv2.imread('s0001_00017_0_0_0_0_0_01.png' , cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(imag_array , cv2.COLOR_GRAY2RGB)
new_array = cv2.resize(backtorgb , (148,148))

x_input = np.array(new_array).reshape(1,148,148,3)
x_input.shape

plt.imshow(new_array)

x_input = x_input/255.0
prediction = history.predict(x_input)

prediction






#lets check on unknown image
#and use opencv to detecte eyes from img
img = cv2.imread('open2.jpg')
plt.imshow(cv2.cvtColor(img , cv2.COLOR_BGR2RGB))

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in eyes:
    cv2.rectangle( img , (x,y) , (x+w,y+h) , (0,255,0) , 2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

#how to crop the eye

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
eyes = eye_cascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in eyes:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h , x:x+w]
    eyes2 = eye_cascade.detectMultiScale(roi_gray)
    if len(eyes2) == 0 :
        print("eyes are not detected")
    else:
        for (x2,y2,w2,h2) in eyes2:
            eyes_roi = roi_color[y2:y2+h2 , x2:x2+w2]
            
plt.imshow(cv2.cvtColor(eyes_roi , cv2.COLOR_BGR2RGB))
    
eyes_roi.shape

final_image = cv2.resize(eyes_roi, (224,224))
final_imahe = np.expand_dims(final_image,axis=0)
final_image = final_image/225.0

final_image.shape
final_image = np.array(new_array).reshape(1,148,148,3)
final_image.shape

pre = history.predict(final_image)



#--------------------------------------------------

#first detect the eyes are closed on open in a face

peth = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#frequency & duration for sound with winsound
frequency = 1000
duration = 1000


#for webcam

cap = cv2.VideoCapture(0)
#check if the webcam is opened correctly

if not cap.isOpened():
    raise IOError("can not open webcam")
    
while True:
    ret,frame = cap.read()
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    eyes = eye_cascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in eyes:
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255,255,255),2)
        eyes2 = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes2) ==0:
            print("eyes are not detected")
        else:
            for (x2,y2,w2,h2) in eyes2:
                eyes_roi = roi_color[y2:y2+h2 , x2:x2+w2]
    
    final_image = cv2.resize(eyes_roi , (148,148))
    final_image = np.expand_dims(final_image , axis =0)
    final_image = final_image/255.0
    
    
    Predictions = history.predict(final_image)
    # i do not know why didn't out 0 or 0< while i use relu
    if(Predictions[0] > 0.5):
        status = "open eyes"
        print(Predictions)
        print(" open ")
    else:
        status = "closed eyes"
        print(Predictions)
        print(" close ")
        winsound.Beep(frequency,duration)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    #draw rectangle aroung the face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h) , (0,0,0),2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #USE PUTtEXT() METHOD FOR
    #insertind text on video
    
    cv2.putText(frame,status,(50,50),font,3,(0,0,255),2,cv2.LINE_4)
        
    cv2.imshow("drowsiness detecction" , frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# if eyes are closed for unusual time , like more than blinks , for few seconds, alarm generated

#i will do it later
    
    
print("kareem") 



