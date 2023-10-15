
from emotion_recognizer import EmotionRecognizer
from emotion_server import EmotionServer
from threading import Thread
from queue import Queue
from config import SERVER_IP, SERVER_PORT, FRAME_RATE, UPDATE_RATE

if __name__ == "__main__":
    data_queue = Queue()
    emotion_server = EmotionServer(SERVER_IP, SERVER_PORT, UPDATE_RATE)
    emotion_recognizer  = EmotionRecognizer(FRAME_RATE)
    update_values_thread = Thread(target=emotion_recognizer.update_values, args=(data_queue,))
    update_to_clients_thread = Thread(target=emotion_server.update_to_clients, args=(data_queue,))
    update_values_thread.start()
    update_to_clients_thread.start()

