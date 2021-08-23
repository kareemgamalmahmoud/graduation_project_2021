from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse

# -----lib for pj ------------------------
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import winsound
import time
import pyttsx3
import threading
import math

import playsound
# import test

# ---------------lib for vol -----------------------
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# -------------------------------------------------


global flag_for_voice

from gtts import gTTS
import os

def help_fun():
    # time.sleep(1)
    # engine = pyttsx3.init()
    # engine.say("First, you must not sleep. Second, avoid taking your eyes off the road. Third, avoid placing your hand close to the eye to avoid increasing blind spots.")
    # engine.runAndWait()

    # tts = gTTS('أولا ، يجب ألا تنام. ثانيًا ، تجنب رفع عينيك عن الطريق. ثالثًا ، تجنب وضع يدك بالقرب من العين لتجنب زيادة النقاط العمياء', lang='ar')
    # tts.save('notes.mp3')
    # os.system("notes.mp3")
    
    playsound.playsound('notes.mp3', True)

    
def alarm_1():
    # winsound.Beep(1000, 100)
    # engine = pyttsx3.init()
    # engine.say("wake up")
    # engine.runAndWait()

    # tts = gTTS(' أستيقظ او خذ أستراحة', lang='ar')
    # tts.save('wake_up.mp3')
    # os.system("wake_up.mp3")

    playsound.playsound('wake_up.mp3', True)

def alarm_for_upper_left_hand():
    # engine = pyttsx3.init()
    # engine.say("Lower your right hand")
    # engine.runAndWait()

    # tts = gTTS('أخفظ يدك اليمنى', lang='ar')
    # tts.save('lower_right.mp3')
    # os.system("lower_right.mp3")

    playsound.playsound('lower_right.mp3', True)

def alarm_for_upper_right_hand():
    # engine = pyttsx3.init()
    # engine.say("Lower your left hand")
    # engine.runAndWait()

    # tts = gTTS('أخفظ يدك اليسرى', lang='ar')
    # tts.save('lower_left.mp3')
    # os.system("lower_left.mp3")

    playsound.playsound('lower_left.mp3', True)

def alarm_for_paying_attention_left():
    # engine = pyttsx3.init()
    # engine.say("Pay attention to the road")
    # engine.runAndWait()

    # tts = gTTS('أنتبه إلى الطريق', lang='ar')
    # tts.save('attention_to_roud.mp3')
    # os.system("attention_to_roud.mp3")

    playsound.playsound('attention_to_roud.mp3', True)



mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# print('------------> ',mp_holistic)
# print('-----------> ',holistic)


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)

    return ear_aspect_ratio

# ---------------this is for vol-----------------------------

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]




# -------------------------------------------------


# DISPLAY CAMERA 1 ------------------

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
def stream_1():
    
    flag_for_voice = True
    p_time = 0
    c_time = 0
    # type of the font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    alarm_flag = False
    alarm_1_var = threading.Thread()



    cam_id = 0
    vid = cv2.VideoCapture(0)
    while True:
        _,image=vid.read()
        image=cv2.resize(image,(900,550))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results=holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # colors
        black = mp_drawing.DrawingSpec(color=(0,0,0) , thickness=1, circle_radius=0)
        white = mp_drawing.DrawingSpec(color=(255,255,255) , thickness=1, circle_radius=0)

        # 1. face landmarks
        mp_drawing.draw_landmarks( image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, black, white)
        # print(results.face_landmarks[0])
        
        if results.face_landmarks :
            left_eye = []
            for id, lm in enumerate(results.face_landmarks.landmark):
                # print(id, lm)
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                
                # left eye
                if id == 33:
                    # print(id, lm)
                    left_eye.append((cx,cy))
                    cv2.circle(image, (cx,cy) ,2,(255,255,150),cv2.FILLED)
                   
                if id == 144:
                    # print(id, lm)
                    left_eye.append((cx,cy))
                    cv2.circle(image, (cx,cy) ,2,(255,255,150),cv2.FILLED)

               
                if id == 153:
                    # print(id, lm)
                    left_eye.append((cx,cy))
                    cv2.circle(image, (cx,cy) ,2,(255,255,150),cv2.FILLED)
                
                if id == 155:
                    # print(id, lm)
                    left_eye.append((cx,cy))
                    cv2.circle(image, (cx,cy) ,2,(255,255,150),cv2.FILLED)
                # ----

                if id == 159:
                    # print(id, lm)
                    left_eye.append((cx,cy))
                    cv2.circle(image, (cx,cy) ,2,(255,255,150),cv2.FILLED)

               
                if id == 158:
                    # print(id, lm)
                    left_eye.append((cx,cy))
                    cv2.circle(image, (cx,cy) ,2,(255,255,150),cv2.FILLED)
                
            left_side = calculate_EAR(left_eye)
            left_side = round(left_side,2)
            print("the ratio : ",left_side)
            if left_side<0.28:
                cv2.putText(image, "closed", (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                if(alarm_flag == False):
                    alarm_1_var = threading.Thread(target=alarm_1)
                    alarm_1_var.start()
                    alarm_flag = True
                # winsound.Beep(1000, 100)
                # alarm_var = threading.Thread(target=alarm_1)
                # alarm_var.start()
            else:
                cv2.putText(image, "open", (50, 50), font, 3, (0, 255, 0), 2, cv2.LINE_4)

            if not alarm_1_var.is_alive():
                alarm_flag = False

        # # 2. pose estimation landmarks
        mp_drawing.draw_landmarks( image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, black, white)
        # print(results.pose_landmarks)
        if results.pose_landmarks:

            y_left_shoulder = 0
            y_right_shoulder = 0
            y_left_wrist = 10
            y_right_wrist = 10

            x_y_left_eye_outer = (0,0)
            x_y_left_ear = (0,0)

            x_y_right_eye_outer = (0,0)
            x_y_right_ear = (0,0)


            for id, lm in enumerate(results.pose_landmarks.landmark):
                # print(id, x)
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
            # ----------------------------------------------------
            # winsound.Beep(1000, 100)



                if id==11:
                    # print(id, cx, cy)
                    y_left_shoulder = cy
                    y_left_wrist = y_left_shoulder + 10
                    cv2.circle(image, (cx,cy) ,10,(255,255,150),cv2.FILLED)
                
                if id==12:
                    # print(id, cx, cy)
                    y_right_shoulder = cy
                    y_right_wrist = y_right_shoulder + 10
                    cv2.circle(image, (cx,cy) ,10,(255,255,150),cv2.FILLED)

                if id==15:
                    # print(id, cx, cy)
                    y_left_wrist = cy
                    # print(cy)
                    cv2.circle(image, (cx,cy) ,10,(255,255,150),cv2.FILLED)

                if id==16:
                    # print(id, cx, cy)
                    y_right_wrist = cy
                    cv2.circle(image, (cx,cy) ,10,(255,255,150),cv2.FILLED)
                
                if y_right_wrist < y_right_shoulder:
                    # winsound.Beep(1000, 100)
                    print("upper_right",y_right_wrist,y_right_shoulder)
                    if(alarm_flag == False):
                        alarm_1_var = threading.Thread(target=alarm_for_upper_right_hand)
                        alarm_1_var.start()
                        alarm_flag = True

                if y_left_wrist < y_left_shoulder:
                    # winsound.Beep(1000, 100)
                    print("upper_left",y_left_wrist,y_left_shoulder)
                    if(alarm_flag == False):
                        alarm_1_var = threading.Thread(target=alarm_for_upper_left_hand)
                        alarm_1_var.start()
                        alarm_flag = True
                    #  cv2.putText(image, "Upper Hands", (30, 150), font, 1, (0, 255, 255), 2, cv2.LINE_4)

                if not alarm_1_var.is_alive() :
                    alarm_flag = False
        #-math.hypot(x2-x1 , y2-y1)------------------note : use math.hypot to get length--------------- يمين و شمال الحساب باستخدام نقطين بس--------------------
                if id==0:
                    # print(id, cx, cy)
                    x_y_left_eye_outer = (cx, cy)
                    cv2.circle(image, (cx,cy) ,10,(255,255,150),cv2.FILLED)
                if id==7:
                    # print(id, cx, cy)
                    x_y_left_ear = (cx,cy)
                    cv2.circle(image, (cx,cy) ,10,(255,255,150),cv2.FILLED)
                if id==8:
                    # print(id, cx, cy)
                    x_y_right_ear = (cx, cy)
                    cv2.circle(image, (cx,cy) ,10,(255,255,150),cv2.FILLED)
                

                # left_distance = (( ((x_y_left_eye_outer[0] - x_y_left_ear[0])**2) - ((x_y_left_eye_outer[1] - x_y_left_ear[1])**2) )**5)
                left_distance = math.hypot(x_y_left_eye_outer[0] - x_y_left_ear[0] , x_y_left_eye_outer[1] - x_y_left_ear[1])
                # print("______>>>>>left_distance",left_distance)
                # left_distance_5d = re.findall("\d{5}", str(left_distance))
                
                # right_distance = (( ((x_y_left_eye_outer[0] - x_y_right_ear[0])**2) - ((x_y_left_eye_outer[1] - x_y_right_ear[1])**2) )**5)
                right_distance = math.hypot(x_y_left_eye_outer[0] - x_y_right_ear[0] , x_y_left_eye_outer[1] - x_y_right_ear[1])
                # print("______>>>>>right_distance",right_distance)
                # right_distance_5d = re.findall("\d{5}", str(right_distance))
                
                if left_distance != 0 :
                    # print(" left_distance - right_distance ",left_distance - right_distance)
                    if left_distance - right_distance > 75 :
                        print("looking_****************************left**************",left_distance - right_distance)
                        if(alarm_flag == False):
                            alarm_1_var = threading.Thread(target=alarm_for_paying_attention_left)
                            alarm_1_var.start()
                            alarm_flag = True
                            print('---------------------->why <--------------------')
                        # winsound.Beep(1000, 100)

                    elif left_distance - right_distance < -500 :
                        print("looking_****************************right**************",left_distance - right_distance)
                        if(alarm_flag == False):
                            print('---------------------->why2 <--------------------')
                            alarm_1_var = threading.Thread(target=alarm_for_paying_attention_left)
                            alarm_1_var.start()
                            alarm_flag = True
                            # winsound.Beep(1000, 100)

                    if not alarm_1_var.is_alive() :
                        alarm_flag = False
                        
                left_distance = 0
                right_distance = 0

            if not alarm_1_var.is_alive() :
                alarm_flag = False
        if not alarm_1_var.is_alive() :
            alarm_flag = False
        
        #         # 3. left hand landmarks    
        mp_drawing.draw_landmarks( image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, black, white)
        # print(results.left_hand_landmarks)

        def findPosition_for_left_hand(img, handNo=0, draw= True):
            
            lmlist = []
            
            if results.left_hand_landmarks:
                
                myHand = results.left_hand_landmarks

                for id , lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(id, cx, cy)
                    l = [id, cx, cy]
                    lmlist.append(l)
                    # if draw:
                    #     cv2.circle(img, (cx,cy) ,5,(0,0,0),cv2.FILLED)
                
            return lmlist

        def findPosition_for_right_hand(img, handNo=0, draw= True):
            
            lmlist = []
            
            if results.right_hand_landmarks:
                
                myHand = results.right_hand_landmarks

                for id , lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(id, cx, cy)
                    l = [id, cx, cy]
                    lmlist.append(l)
                    # if draw:
                    #     cv2.circle(img, (cx,cy) ,5,(0,0,0),cv2.FILLED)
                
            return lmlist

        # 7ttet al threading -->done
        left_hand_position = findPosition_for_left_hand(image)
        if len(left_hand_position) != 0:
            if left_hand_position[8][2] < left_hand_position[6][2] and left_hand_position[12][2] > left_hand_position[10][2]:
                print("first_finger",left_hand_position[8][2],left_hand_position[6][2] ,"sec_finger " ,left_hand_position[12][2] ,left_hand_position[10][2])
            
                # if flag_for_voice:
                #     flag_for_voice = False
                #     help_var = threading.Thread(target=help_fun)
                #     help_var.start()
                if(alarm_flag == False):
                    alarm_1_var = threading.Thread(target=help_fun)
                    alarm_1_var.start()
                    alarm_flag = True


            elif left_hand_position[8][2] < left_hand_position[6][2] and left_hand_position[12][2] < left_hand_position[10][2] and left_hand_position[16][2] > left_hand_position[14][2]:
                # 4. right hand landmarks
                mp_drawing.draw_landmarks( image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, black, white)
                right_hand_position = findPosition_for_right_hand(image)
                if len(right_hand_position) != 0:
                    # print(right_hand_position[4] , right_hand_position[8])

                    x1 , y1 = right_hand_position[4][1] , right_hand_position[4][2]
                    x2 , y2 = right_hand_position[8][1] , right_hand_position[8][2]
                    cx , cy = (x1+x2)//2 , (y1+y2)//2


                    cv2.circle(image, (x1,y1) ,10,(255,255,150),cv2.FILLED)
                    cv2.circle(image, (x2,y2) ,10,(255,255,150),cv2.FILLED)
                    cv2.line(image , (x1,y1),(x2,y2),(255,255,100),2)
                    cv2.circle(image, (cx,cy) ,10,(255,255,150),cv2.FILLED)

                    length = math.hypot(x2-x1 , y2-y1)
                    print(length)

                    vol = np.interp(length , [0,100] , [min_vol+50,max_vol])
                    print(length , vol)
                    volume.SetMasterVolumeLevel(vol, None)

                    if length < 30:
                        cv2.circle(image, (cx,cy) ,10,(0,0,0),cv2.FILLED)


            else:
                flag_for_voice = True
            
            if not alarm_1_var.is_alive() :
                alarm_flag = False

        

        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time

        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)




        image = cv2.putText(np.float32(image),"hello",(50,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)    
        cv2.imwrite('demo.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')


def video_feed_1(request):
        return StreamingHttpResponse(stream_1(), content_type='multipart/x-mixed-replace; boundary=frame')
# -----------------------------------



# HOME PAGE -------------------------
def test(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render({}, request))
    # return render(request, 'index.html', context={'text':'hello world'})
# -----------------------------------
