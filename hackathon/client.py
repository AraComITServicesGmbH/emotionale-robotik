from threading import Thread
from queue import Queue
import time

from config import SERVER_IP, SERVER_PORT
from communication import create_client_socket, receive_image_and_emotions
from robot import Robot
from own_robot import OwnRobot

from cli import print_connect, print_disconnect


def update_robot(queue: Queue, robot: Robot):
    while True:
        photo, emotions = queue.get()
        with queue.mutex:
            queue.queue.clear()
        robot.update(photo, emotions)
        time.sleep(robot.update_interval)


def update_from_server(queue: Queue):
    while True:
        client_socket = create_client_socket()
        try:
            client_socket.connect((SERVER_IP, SERVER_PORT))
            print_connect()
            while True:
                try:
                    result = receive_image_and_emotions(client_socket)
                    if result is not None:
                        queue.put(result)
                except EOFError:
                    pass
        except Exception:
            print_disconnect()
            client_socket.close()
    time.sleep(1)


if __name__ == "__main__":
    data_queue = Queue()
    own_robot = OwnRobot()
    update_from_server_thread = Thread(target=update_from_server, args=(data_queue,))
    update_robot_thread = Thread(target=update_robot, args=(data_queue, own_robot))
    update_from_server_thread.start()
    update_robot_thread.start()
