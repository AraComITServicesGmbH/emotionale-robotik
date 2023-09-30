import socket
import cv2 as cv
import pickle
from pickle import UnpicklingError


def create_client_socket():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10000000)
    return client_socket


def create_server_socket(ip, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    return server_socket


def send_image_and_emotions(open_socket, address, photo, emotions):
    ret, buffer = cv.imencode(".jpg", photo, [int(cv.IMWRITE_JPEG_QUALITY), 30])
    x_as_bytes = pickle.dumps((buffer, emotions))
    open_socket.sendall(x_as_bytes)

    


def receive_image_and_emotions(open_socket):
    try:
        x = open_socket.recvfrom(1000000)
        data = x[0]
        (data, emotions) = pickle.loads(data)
        photo = cv.imdecode(data, cv.IMREAD_COLOR)
        return (photo, emotions)
    except UnpicklingError:
        pass
    return None
