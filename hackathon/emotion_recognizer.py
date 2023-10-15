from queue import Queue
import time
import torch
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

class EmotionRecognizer:
    def __init__(self, frame_rate):
        self.frame_rate = frame_rate
        self.check_and_print_cuda_available()
        self.mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=self.device)
        self.emotion_recognizer = HSEmotionRecognizer(
            model_name="enet_b0_8_best_afew", device=self.device
        )
        self.device = "cpu"
        
    def check_and_print_cuda_available(self):
        """
        Checks if CUDA is available using PyTorch and prints a corresponding message.
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("Cuda will be used for calculation")
            self.device = "cuda"
        else:
            print("Warning: CPU will be used for calculation.")
            print("The calculation could be slowly.")
            self.device = "cpu"


    def ensure_camera_available(self):
        """
        This function checks the status of a given camera capture object. If the camera
        is not open or available, it prints an error message and terminates the program.
        """
        if not self.capture.isOpened():
            print("Cannot open camera")
            exit()


    def detect_and_filter_faces(self, probability_threshold=0.9):
        """
        Detects faces in a given frame using the MTCNN detector and filters them based on a probability threshold.

        This function first detects faces in the frame with their associated probabilities using the MTCNN model.
        Then, it filters out bounding boxes whose associated probabilities are below a specified threshold.

        Parameters:
        - threshold (float, optional): The probability threshold below which bounding boxes are filtered out.
                                    Default is 0.9.

        Returns:
        - list of lists: A list of bounding boxes that passed the threshold. Each bounding box is represented 
                        as a list of four numbers: [x1, y1, x2, y2], where (x1, y1) is the top-left corner 
                        and (x2, y2) is the bottom-right corner of the bounding box.
    ^   """
        self.bounding_boxes, probabilities = self.mtcnn.detect(self.frame, landmarks=False)
        if self.bounding_boxes is not None:
            self.bounding_boxes = self.bounding_boxes[probabilities > probability_threshold]
        else:
            self.bounding_boxes = []
        


    def sort_values_from_bounding_boxes(self):
        """
        Sorts bounding box values to ensure they are within specified limits.
        The function ensures x1 < x2 and y1 < y2.
        """
        sorted_bounding_boxes = []
        for bounding_box in self.bounding_boxes:
            box = bounding_box.astype(int)
            x1, y1, x2, y2 = box[0:4]
            xs = [x1, x2]
            ys = [y1, y2]
            xs.sort()
            ys.sort()
            sorted_box = np.array([xs[0], ys[0], xs[1], ys[1]])
            sorted_bounding_boxes.append(sorted_box)
        self.bounding_boxes = sorted_bounding_boxes


    def clip_values_from_bounding_boxes(self):
        """
        Clips bounding box values to ensure they are within specified limits.
        """
        x_limit = self.frame.shape[0]
        y_limit = self.frame.shape[1]
        limited_bounding_boxes = []
        for bounding_box in self.bounding_boxes:
            box = bounding_box.astype(int)
            x1, y1, x2, y2 = box[0:4]
            x1 = np.clip(x1, 0, x_limit - 2)
            x2 = np.clip(x2, 0, x_limit - 1)
            y1 = np.clip(y1, 0, y_limit - 2)
            y2 = np.clip(y2, 0, y_limit - 1)
            limited_box = np.array([x1, y1, x2, y2])
            limited_bounding_boxes.append(limited_box)
        self.bounding_boxes = limited_bounding_boxes


    def predict_emotions_from_faces(self):
        """
        Given an image frame and bounding boxes corresponding to detected faces, this function extracts each face 
        and uses the provided emotion recognizer to predict the emotion exhibited by each face.
        """
        emotions = []
        for bounding_box in self.bounding_boxes:
            x1, y1, x2, y2 = bounding_box[0:4]
            face_img = self.frame[y1:y2, x1:x2, :]
            emotion, _ = self.emotion_recognizer.predict_emotions(face_img, logits=True)
            emotions.append(emotion)
        self.emotions = emotions
    
    def update_values(self, queue: Queue):    
        """
        Continuously captures video frames from the default camera and 
        detects faces along with their associated emotions in each frame. The processed
        frame and emotions are then added to the given queue. If the queue is full,
        the previous contents are cleared.
        
        The function uses an MTCNN detector for face detection and the HSEmotionRecognizer 
        for emotion recognition. Frames are processed at a specified frame rate.
        
        Parameters:
        - queue (Queue): A queue object to store the captured frame and the detected emotions.
        """
        previous_time = 0
        self.capture = cv.VideoCapture(0)
        self.ensure_camera_available()
        try:
            while True:
                time_elapsed = time.time() - previous_time
                if time_elapsed > 1/self.frame_rate:
                    previous_time = time.time()
                    _, frame_bgr = self.capture.read()
                    self.frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                    self.detect_and_filter_faces()
                    self.sort_values_from_bounding_boxes()
                    self.clip_values_from_bounding_boxes()
                    self.predict_emotions_from_faces()
                    if queue.full():
                        with queue.mutex:
                            queue.queue.clear()
                            queue.put((frame_bgr, self.emotions))
                    else:
                        queue.put((frame_bgr, self.emotions))
        finally:
            self.capture.release()