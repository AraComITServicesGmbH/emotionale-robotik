import cv2 as cv
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from threading import Thread
from queue import Queue
import time
from config import SERVER_IP, SERVER_PORT, FRAME_RATE, UPDATE_RATE
from communication import create_server_socket, send_image_and_emotions
from vision import (
    check_and_print_cuda_available,
    ensure_camera_available,
    detect_and_filter_faces,
    sort_and_clip_values_from_bounding_boxes,
    predict_emotions_from_faces,
)
from cli import print_connect, print_disconnect, print_emotions


def update_values(queue: Queue):    
    """
    Continuously captures video frames from the default camera and 
    detects faces along with their associated emotions in each frame. The processed
    frame and emotions are then added to the given queue. If the queue is full,
    the previous contents are cleared.
    
    The function uses an MTCNN detector for face detection and the HSEmotionRecognizer 
    for emotion recognition. Frames are processed at a specified FRAME_RATE.
    
    Parameters:
    - queue (Queue): A queue object to store the captured frame and the detected emotions.
    """
    capture = cv.VideoCapture(0)
    ensure_camera_available(capture)
    device = check_and_print_cuda_available()
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
                bounding_boxes = sort_and_clip_values_from_bounding_boxes(
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
    """
    Listens for incoming client connections and spawns a new thread 
    to handle updates for each connected client.
    
    The function establishes a server socket and listens for incoming client 
    connections on the specified SERVER_IP and SERVER_PORT. 

    The server socket will continue to listen for connections until an external 
    interruption. On termination, the server socket is closed gracefully.

    Parameters:
    - queue (Queue): A queue object to facilitate inter-thread communication.
    """
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
    """
    Sends updates to a client at a specified rate defined by UPRDATE_RATE 
    using frame in BGR format and emotions from a queue.

    The function continues to serve the client indefinitely until an exception 
    occurs, such as a connection disruption. Finally it notifies about the client's
    disconnection and closes the client socket.

    Parameters:
    - client (socket): The socket object representing the client connection.
    - address (tuple): A tuple containing the client's IP address and port number.
    - queue (Queue): A queue object from which frames (in BGR format) and emotions are fetched.
    """
    try:
        previous_time = 0
        print_connect(address)
        while True:
            time_elapsed = time.time() - previous_time
            if time_elapsed > 1/UPDATE_RATE:
                time_elapsed = time.time()
                (frame_bgr, emotions) = queue.get()
                send_image_and_emotions(client, frame_bgr, emotions)
    except Exception:
        print_disconnect(address)
        client.close()


if __name__ == "__main__":
    data_queue = Queue()
    update_values_thread = Thread(target=update_values, args=(data_queue,))
    update_to_clients_thread = Thread(target=update_to_clients, args=(data_queue,))
    update_values_thread.start()
    update_to_clients_thread.start()
