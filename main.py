import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('gf/clip4.mp4')
color = (0, 255, 0)
text = 'Loyal GF'
while True:
    _, frame = cap.read()
    objects = model.predict(frame, verbose=False)
    people = []
    for object in objects:
        for box in object.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                people.append(x1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if len(people) == 2: 
        distance = abs(people[0] - people[1])
        if distance < 100: 
            color = (0, 0, 255)
            text = 'Cheater GF'

    cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.imshow('Cheat Detector', frame)
    if cv2.waitKey(10) == ord('q'):
        break
