import cv2
import numpy as np
from fer import FER
from logger import log_detection
from datetime import datetime
import subprocess

# Load pre-trained models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
           '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
emotion_detector = FER()

def getFaceBox(net, frame, conf_threshold=0.75):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

cap = cv2.VideoCapture(0)

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    bboxes = getFaceBox(faceNet, small_frame)

    for bbox in bboxes:
        face = small_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]

        emotion_predictions = emotion_detector.detect_emotions(face)
        emotion = max(emotion_predictions[0]['emotions'],
                      key=emotion_predictions[0]['emotions'].get) if emotion_predictions else "Unknown"

        label = f"{gender}, {age}, {emotion}"
        cv2.rectangle(frame, (bbox[0]*2, bbox[1]*2),
                      (bbox[2]*2, bbox[3]*2), (0, 255, 0), 2)
        cv2.putText(frame, label, (bbox[0]*2, bbox[1]*2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Save to CSV
        log_detection(datetime.now(), gender, age, emotion)

    cv2.imshow("Age, Gender, Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
subprocess.run(["python", "report_emailer.py"])