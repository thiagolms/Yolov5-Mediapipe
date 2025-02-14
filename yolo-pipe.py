import cv2
import torch
import mediapipe as mp
import json

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model.classes = [0]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

video_path = "br-07.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video_yolo-pipe.mp4", fourcc, fps, (width, height))

pose_data = {"frames": []}
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)
    people_boxes = []
    
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:
            people_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    keypoints_per_person = []

    for (x1, y1, x2, y2) in people_boxes:
        person_roi = frame_rgb[y1:y2, x1:x2]

        if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
            results_pose = pose.process(person_roi)

            if results_pose.pose_landmarks:
                keypoints = []
                for landmark in results_pose.pose_landmarks.landmark:
                    keypoints.append(x1 + int(landmark.x * (x2 - x1)))
                    keypoints.append(y1 + int(landmark.y * (y2 - y1)))

                keypoints_per_person.append(keypoints)

                mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    out.write(frame)
    frame_index += 1

cap.release()
out.release()

print("Vídeo salvo")
