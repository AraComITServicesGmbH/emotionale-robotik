from threading import Thread
from queue import Queue

from config import SERVER_IP, SERVER_PORT
from emotion_client import EmotionClient
from own_robot import OwnRobot


if __name__ == "__main__":
    data_queue = Queue()
    own_robot = OwnRobot()
    emotion_client = EmotionClient(SERVER_IP, SERVER_PORT)
    update_from_server_thread = Thread(target=emotion_client.update_from_server, args=(data_queue,))
    update_robot_thread = Thread(target=own_robot.update_from_queue, args=(data_queue,))
    update_from_server_thread.start()
    update_robot_thread.start()
