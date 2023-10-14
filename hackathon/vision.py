import torch
import numpy as np

def check_and_print_cuda_available():
    """
    Checks if CUDA is available using PyTorch and prints a corresponding message.

    Returns:
        str: "cuda" if CUDA is available, otherwise "cpu".
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Cuda will be used for calculation")
        return "cuda"
    else:
        print("Warning: CPU will be used for calculation.")
        print("The calculation could be slowly.")
        return "cpu"


def ensure_camera_available(capture):
    """
    This function checks the status of a given camera capture object. If the camera
    is not open or available, it prints an error message and terminates the program.

    Parameters:
        capture: A capture object from OpenCV representing the camera feed.
    """
    if not capture.isOpened():
        print("Cannot open camera")
        exit()


def detect_and_filter_faces(mtcnn, frame, probability_threshold=0.9):
    """
    Detects faces in a given frame using the MTCNN detector and filters them based on a probability threshold.

    This function first detects faces in the frame with their associated probabilities using the MTCNN model.
    Then, it filters out bounding boxes whose associated probabilities are below a specified threshold.

    Parameters:
    - frame (numpy.ndarray): A numpy array representing the image in which faces need to be detected.
    - mtcnn (MTCNN object): An instance of the MTCNN detector used for face detection.
    - threshold (float, optional): The probability threshold below which bounding boxes are filtered out.
                                   Default is 0.9.

    Returns:
    - list of lists: A list of bounding boxes that passed the threshold. Each bounding box is represented 
                     as a list of four numbers: [x1, y1, x2, y2], where (x1, y1) is the top-left corner 
                     and (x2, y2) is the bottom-right corner of the bounding box.
^   """
    bounding_boxes, probabilities = mtcnn.detect(frame, landmarks=False)
    if bounding_boxes is not None:
        bounding_boxes = bounding_boxes[probabilities > probability_threshold]
    else:
        bounding_boxes = []
    return bounding_boxes


def sort_and_clip_values_from_bounding_boxes(bounding_boxes, x_limit, y_limit):
    """
    Sorts and clips bounding box values to ensure they are within specified limits.

    Given a list of bounding boxes, this function:
    1. Sorts the x and y coordinates within each bounding box to ensure x1 < x2 and y1 < y2.
    2. Clips the x and y values of each bounding box to be within the specified limits.

    Parameters:
    - bounding_boxes (list of lists): A list of bounding boxes. Each bounding box should be a list 
                                     of four numbers in the format: [x1, y1, x2, y2].
    - x_limit (int): The maximum allowed value for the x-coordinates. Values above this will be clipped.
    - y_limit (int): The maximum allowed value for the y-coordinates. Values above this will be clipped.

    Returns:
    - list of lists: A list of sorted and clipped bounding boxes. Each bounding box is in the format: [x1, y1, x2, y2].
    """
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
    """
    Given an image frame and bounding boxes corresponding to detected faces, this function extracts each face 
    and uses the provided emotion recognizer to predict the emotion exhibited by each face.

    Parameters:
    - emotion_recognizer (Object): An object responsible for recognizing and predicting emotions
    - frame (numpy.ndarray): A numpy array representing the image frame in which emotions need to be predicted.
    - bounding_boxes (list of lists): A list of bounding boxes. Each bounding box should be a list 
                                     of four numbers in the format: [x1, y1, x2, y2].

    Returns:
    - list: A list of predicted emotions corresponding to each bounding box.
    """
    emotions = []
    for bounding_box in bounding_boxes:
        x1, y1, x2, y2 = bounding_box[0:4]
        face_img = frame[y1:y2, x1:x2, :]
        emotion, _ = emotion_recognizer.predict_emotions(face_img, logits=True)
        emotions.append(emotion)
    return emotions