import torch
import numpy as np

def check_cuda_available():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Cuda will be used for calculation")
        return "cuda"
    else:
        print("Warning: CPU will be used for calculation.")
        print("The calculation could be slowly.")
        return "cpu"


def check_camera_available(capture):
    if not capture.isOpened():
        print("Cannot open camera")
        exit()


def detect_faces(mtcnn, frame):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    if bounding_boxes is not None:
        bounding_boxes = bounding_boxes[probs > 0.9]
    else:
        bounding_boxes = []
    return bounding_boxes


def adjust_bounding_boxes(bounding_boxes, x_limit, y_limit):
    limited_bounding_boxes = []
    for bounding_box in bounding_boxes:
        box = bounding_box.astype(int)
        x1, y1, x2, y2 = box[0:4]
        xs = [x1, x2]
        ys = [y1, y2]
        xs.sort()
        ys.sort()
        xs[0] = np.clip(xs[0], 0, x_limit - 2)
        xs[1] = np.clip(xs[1], 0, x_limit - 1)
        ys[0] = np.clip(ys[0], 0, y_limit - 2)
        ys[1] = np.clip(ys[1], 0, y_limit - 1)

        limited_box = [xs[0], ys[0], xs[1], ys[1]]
        limited_bounding_boxes.append(limited_box)
    return limited_bounding_boxes


def predict_emotions_from_faces(emotion_recognizer, frame, bounding_boxes):
    emotions = []
    for bounding_box in bounding_boxes:
        x1, y1, x2, y2 = bounding_box[0:4]
        face_img = frame[y1:y2, x1:x2, :]
        emotion, _ = emotion_recognizer.predict_emotions(face_img, logits=True)
        emotions.append(emotion)
    return emotions