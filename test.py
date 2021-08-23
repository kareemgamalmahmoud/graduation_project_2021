import mediapipe as mp
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("any_thing")
print(mp_holistic)
print(holistic)

