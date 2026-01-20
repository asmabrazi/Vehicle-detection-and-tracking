import cv2
from ultralytics import YOLO
from collections import defaultdict
import torch

model = YOLO('yolo11l.pt')

nb_classes = len(model.names)
class_colors = defaultdict()
for i in range(nb_classes):
    #class_colors[i] = (int(torch.randint(0, 255, (1,)).item()), int(torch.randint(0, 255, (1,)).item()), int(torch.randint(0, 255, (1,)).item()))
    class_colors[1] = (255, 0, 0)   # bicycle - blue
    class_colors[2] = (0, 255, 0)   # car - green
    class_colors[3] = (255, 0, 255) # motorcycle - magenta
    class_colors[5] = (0, 255, 100) # bus - cyan
    class_colors[7] = (0, 165, 255) # truck - orange

cap = cv2.VideoCapture("tf.mp4")
my_counter_above = {}
my_counter_below = {}
all_detected_objects = {}

coming_from_above_limity = 200
coming_from_above_limitx = 705
coming_from_below_limity = 600

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, device="mps")

    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the detected boxes, their class indices and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        #loop over each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            if class_idx not in {1, 2, 3, 5, 7}: 
                continue 
            #if conf < 0.3:
            #    continue

            x1, y1, x2, y2 = torch.round(box).int().tolist()
            bb_centerx, bb_centery = ((x1 + x2) // 2, (y1 + y2) // 2)
            label = f"{model.names[class_idx]} ID:{track_id} {conf:.2f}"
            color = class_colors[class_idx]
            if track_id not in all_detected_objects:
                all_detected_objects[track_id] = {
                    "class_idx": class_idx,
                    "coming_from_above": True if bb_centery < coming_from_above_limity else False,
                    "counted": False
                }

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (bb_centerx, bb_centery), 5, color, -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Check if the object has crossed the line
            if all_detected_objects[track_id]["coming_from_above"] and bb_centery >= coming_from_above_limity and not all_detected_objects[track_id]["counted"]:
                all_detected_objects[track_id]["counted"] = True
                if class_idx in my_counter_above:
                    my_counter_above[class_idx] += 1
                else:
                    my_counter_above[class_idx] = 1

            elif not all_detected_objects[track_id]["coming_from_above"] and bb_centerx >= coming_from_above_limitx and bb_centery <= coming_from_below_limity and not all_detected_objects[track_id]["counted"]:
                all_detected_objects[track_id]["counted"] = True
                if class_idx in my_counter_below:
                    my_counter_below[class_idx] += 1
                else:
                    my_counter_below[class_idx] = 1
            
        i = 0
        for key, value in my_counter_above.items():
            count_label = f"{model.names[key]}: {value}"
            cv2.putText(frame, count_label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            i += 1

        i = 0
        for key, value in my_counter_below.items():
            count_label = f"{model.names[key]}: {value}"
            cv2.putText(frame, count_label, (1800, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
            i += 1

    cv2.line(frame, (455, coming_from_above_limity), (720, coming_from_above_limity), (0, 0, 255), 3)
    cv2.line(frame, (coming_from_above_limitx, coming_from_below_limity), (1400, coming_from_below_limity), (0, 255, 255), 3)
    cv2.imshow("YOLO11 Object Detection and Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()