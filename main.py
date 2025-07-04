import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('gf/apt2.mp4')
color = (0, 255, 0)
text = 'Loyal GF'
while True:
    _, frame = cap.read()
    objects = model.predict(source=frame, verbose = False)
    people = []
    for objects in objects:
        for box in objects.boxes:
            if int(box.cls[0] == 0):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                people.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 )
    
    if len(people) == 2:
        x1a, y1a, x2a, y2a = people[0]
        x1b, y1b, x2b, y2b = people[1]
        if abs(x1a-x1b) < 100:
            color=(0,0,255)
            text='Cheating gf'
    cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)
    cv2.imshow('Cheat Detection', frame)
    if cv2.waitKey(10) == ord('q'):
        break