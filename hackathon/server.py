import cv2 as cv
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from threading import Thread
from queue import Queue
import time
from config import SERVER_IP, SERVER_PORT, FRAME_RATE, UPDATE_RATE
from communication import create_server_socket, send_image_and_emotions
from vision import (
    check_cuda_available,
    check_camera_available,
    detect_and_filter_faces,
    adjust_bounding_boxes,
    predict_emotions_from_faces,
)
from cli import print_connect, print_disconnect, print_emotions


def update_values(queue: Queue):
    capture = cv.VideoCapture(0)
    check_camera_available(capture)
    device = check_cuda_available()
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
    emotion_recognizer = HSEmotionRecognizer(
        model_name="enet_b0_8_best_afew", device=device
    )
    previous_time = 0
    try:
        while True:
            time_elapsed = time.time() - previous_time
            if time_elapsed > 1/FRAME_RATE:
                previous_time = time.time()
                _, frame_bgr = capture.read()
                frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                bounding_boxes = detect_and_filter_faces(mtcnn, frame)
                bounding_boxes = adjust_bounding_boxes(
                    bounding_boxes, frame.shape[0], frame.shape[1]
                )
                emotions = predict_emotions_from_faces(
                    emotion_recognizer, frame, bounding_boxes
                )
                #print_emotions(emotions)
                if queue.full():
                    with queue.mutex:
                        queue.queue.clear()
                        queue.put((frame_bgr, emotions))
                else:
                    queue.put((frame_bgr, emotions))
    finally:
        capture.release()


def update_to_clients(queue: Queue):
    server_socket = create_server_socket(SERVER_IP, SERVER_PORT)
    
    try:
        server_socket.listen(5)
        while True:
            client, address = server_socket.accept()
            thread = Thread(
                target=update_to_client, args=(client, address, queue)
            )
            thread.start()
    finally:
        server_socket.close()


def update_to_client(client, address, queue):
    
    try:
        previous_time = 0
        print_connect(address)
        while True:
            time_elapsed = time.time() - previous_time
            if time_elapsed > 1/UPDATE_RATE:
                time_elapsed = time.time()
                (frame_bgr, emotions) = queue.get()
                send_image_and_emotions(client, address, frame_bgr, emotions)
    except Exception:
        print_disconnect(address)
        client.close()


if __name__ == "__main__":
    data_queue = Queue()
    update_values_thread = Thread(target=update_values, args=(data_queue,))
    update_to_clients_thread = Thread(target=update_to_clients, args=(data_queue,))
    update_values_thread.start()
    update_to_clients_thread.start()
