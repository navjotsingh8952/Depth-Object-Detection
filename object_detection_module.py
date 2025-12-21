import cv2

with open("./res/coco.names", "rt") as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = './res/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = './res/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def detect(img):
    classIds, confs, bbox = net.detect(img, confThreshold=0.45)
    if len(classIds) == 0:
        return None

    objects_detected = []
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        object_detected = classNames[classId - 1]
        objects_detected.append((object_detected, box, confidence))
    return objects_detected


def detect_object(frame):
    detected_objects = detect(frame)
    if detected_objects is None:
        return frame, None
    objects = []
    for object, bounding_box, confidence in detected_objects:
        cv2.rectangle(frame, bounding_box, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, object.upper(), (bounding_box[0] + 10, bounding_box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(round(confidence * 100, 2)), (bounding_box[0] + 200, bounding_box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        objects.append(object)
    return frame, objects
