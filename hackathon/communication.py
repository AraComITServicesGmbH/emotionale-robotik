import socket
import cv2 as cv
import pickle
from pickle import UnpicklingError


def create_client_socket():
    """
    Creates and configures a client socket for TCP communication.
    
    This function initializes a new client socket using the IPv4 address family 
    and the TCP protocol. It also configures the socket's send buffer size 
    to 100,000 bytes to optimize sending performance.

    Returns:
        socket: A configured TCP client socket using IPv4.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 100000)
    return client_socket


def create_server_socket(ip, port):
    """
    Creates a server socket bound to the specified IP address and port.

    This function initializes a new server socket using the IPv4 address family
    and the TCP protocol. It then binds the socket to the given IP address and port.

    Parameters:
    - ip (str): The IP address to which the server socket should be bound.
    - port (int): The port number to which the server socket should be bound.

    Returns:
        socket: A bound TCP server socket using IPv4.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    return server_socket


def send_image_and_emotions(open_socket, photo, emotions):
    """Sends an image and emotions over a socket.

    Parameters:
    - open_socket (socket.socket): An open socket over which the data should be sent. 
    - photo (numpy.ndarray): A numpy array representing the image to be sent.
    - emotions (list|tuple): A list containing the emotions associated with the photo.

    Raises:
        Exception: If an error occurs while sending the data.
    """
    ret, buffer = cv.imencode(".jpg", photo, [int(cv.IMWRITE_JPEG_QUALITY), 30])
    x_as_bytes = pickle.dumps((buffer, emotions))
    open_socket.sendall(x_as_bytes)


def receive_image_and_emotions(open_socket):
    """
    Receives an image and emotions from an open socket.
    
    Args:
        open_socket: A TCP socket that is open and connected to the sender.

    Returns:
        A tuple containing the received image and emotions, or `None` if an error occurred.

    Raises:
        UnpicklingError: If the received data cannot be unpickled.
    """
    try:
        x = open_socket.recvfrom(100000)
        data = x[0]
        (data, emotions) = pickle.loads(data)
        photo = cv.imdecode(data, cv.IMREAD_COLOR)
        return (photo, emotions)
    except UnpicklingError:
        pass
    return None
