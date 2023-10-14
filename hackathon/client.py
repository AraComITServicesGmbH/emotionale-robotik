from threading import Thread
from queue import Queue
import time

from config import SERVER_IP, SERVER_PORT
from communication import create_client_socket, receive_image_and_emotions
from robot import Robot
from own_robot import OwnRobot

from cli import print_connect, print_disconnect


def update_robot(queue: Queue, robot: Robot):
    """
    Continuously update the robot with photos and corresponding emotions 
    from the queue at the robot's specified update interval. After processing 
    an item from the queue, it clears the rest of the queue to ensure the 
    robot only receives the latest information.

    Parameters:
    - queue (Queue): A queue containing tuples of photos and corresponding emotions.
    - robot (Robot): An instance of a Robot that has an update method and an update_interval attribute.

    Note:
    This function runs in an infinite loop and should typically be executed in a separate thread.
    """
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
