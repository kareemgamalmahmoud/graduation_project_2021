"""
@author: kareem
"""

import cv2 as cv
import matplotlib.pyplot as plt
import winsound #this works only for windows

net = cv.dnn.readNetFromTensorflow("graph_opt.pb") ##the waights

inWidth = 368
inHeight = 360
thr = 0.2

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

img = cv.imread("standman.jpg")## note : bgr in colors 
plt.imshow(cv.cvtColor(img , cv.COLOR_BGR2RGB))

def pose_estimation(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inp = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    assert(len(BODY_PARTS) <= out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            
            print(partFrom , "  " ,points[idFrom][1])
            print(partTo , "  " ,points[idTo][1])
            print(" ")
            
            cv.line(frame, points[idFrom], points[idTo], (0, 0, 0), 1)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (255, 255, 255), cv.FILLED)
    
            cv.putText(frame, 'Down Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0),2)
            
            
            if (partTo == "RWrist"):
                if points.__contains__(points[BODY_PARTS["RShoulder"]]):
                    if (points[BODY_PARTS["RShoulder"]][1] < int(points[BODY_PARTS["RWrist"]][1]) ):
                        #print(" the wrist is down")
                        cv.putText(frame, 'Down Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0),2)
                    else:
                        cv.putText(frame, ' upper Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200),2)
            elif (partTo == "LWrist"):
                if points.__contains__(points[BODY_PARTS["LShoulder"]]):
                    if points[BODY_PARTS["LShoulder"]][1] < int(points[BODY_PARTS["LWrist"]][1]):
                        #print(" the wrist is down")
                        cv.putText(frame, 'Down Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0),2)
                    else:
                        cv.putText(frame, ' upper Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200),2)
                            
            else:
                cv.putText(frame, 'Down Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0),2)
    
    if (points[BODY_PARTS["REar"]] and points[BODY_PARTS["LEar"]]):
        cv.putText(frame, 'looking ahead' , (5, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0),2)               
    elif (points[BODY_PARTS["REar"]]):
        cv.putText(frame, 'looking left' , (5, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200),2)              
    elif (points[BODY_PARTS["LEar"]]):
        cv.putText(frame, 'looking right' , (5, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200),2)    
            
    #t, _ = net.getPerfProfile()
    #freq = cv.getTickFrequency() / 1000
    #cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
    return frame

estamed_image = pose_estimation(img)
plt.imshow(cv.cvtColor(estamed_image , cv.COLOR_BGR2RGB))


#------------------------
# perform this demo on webcam
#------------------------


cap = cv.VideoCapture(1)

cap.set(cv.CAP_PROP_FPS,10)
cap.set(cv.CAP_PROP_FRAME_WIDTH,2000)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,2000)

if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcame")

while True:
    
    if cv.waitKey(2) & 0xFF == ord('q'):
        break;
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inp = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    assert(len(BODY_PARTS) <= out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            
            print(partFrom , "  " ,points[idFrom][1])
            print(partTo , "  " ,points[idTo][1])
            print(" ")
            
            cv.line(frame, points[idFrom], points[idTo], (0, 0, 0), 1)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (255, 255, 255), cv.FILLED)
            
            
            
            if (partTo == "RWrist"):
                if points.__contains__(points[BODY_PARTS["RShoulder"]]):
                    if (points[BODY_PARTS["RShoulder"]][1] < int(points[BODY_PARTS["RWrist"]][1]) ):
                        #print(" the wrist is down")
                        cv.putText(frame, 'Down Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0),2)
                    else:
                        cv.putText(frame, ' upper Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200),2)
                        winsound.Beep(1000,1000)
            elif (partTo == "LWrist"):
                if points.__contains__(points[BODY_PARTS["LShoulder"]]):
                    if points[BODY_PARTS["LShoulder"]][1] < int(points[BODY_PARTS["LWrist"]][1]):
                        #print(" the wrist is down")
                        cv.putText(frame, 'Down Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0),2)
                    else:
                        cv.putText(frame, ' upper Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200),2)
                        winsound.Beep(1000,1000)
            #else:
             #   cv.putText(frame, 'Down Hand' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0),2)
    
    if (points[BODY_PARTS["REar"]] and points[BODY_PARTS["LEar"]]):
        cv.putText(frame, 'looking ahead' , (5, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0),2)               
    elif (points[BODY_PARTS["REar"]]):
        cv.putText(frame, 'looking left' , (5, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200),2) 
        winsound.Beep(1000,1000)              
    elif (points[BODY_PARTS["LEar"]]):
        cv.putText(frame, 'looking right' , (5, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200),2)   
        winsound.Beep(1000,1000)            
            
    #t, _ = net.getPerfProfile()
    #freq = cv.getTickFrequency() / 1000
    #cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
    cv.imshow('OpenPose using OpenCV', frame)


cap.release()
cv.destroyAllWindows()

