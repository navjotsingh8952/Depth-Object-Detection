import cv2

with open("./res/coco.names", "rt") as f:
    classNames = f.read().rstrip('\n').split('\n')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

configPath = './res/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = './res/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def detect_object(img):
    classIds, confs, bbox = net.detect(img, confThreshold=0.45)
    if len(classIds) == 0:
        return None

    objects_detected = []
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        object_detected = classNames[classId - 1]
        objects_detected.append((object_detected, box, confidence))
    return objects_detected


def detect():
    while True:
        success, img = cap.read()
        if not success:
            continue

        detected_objects = detect_object(img)
        if detected_objects is None:
            continue

        for object, bounding_box, confidence in detected_objects:
            cv2.rectangle(img, bounding_box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, object.upper(), (bounding_box[0] + 10, bounding_box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (bounding_box[0] + 200, bounding_box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            print(object)
        cv2.imshow("Output", img)
        cv2.waitKey(3)


if __name__ == '__main__':
    detect()
