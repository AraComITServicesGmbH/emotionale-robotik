import cv2 as cv
import pickle
from pickle import UnpicklingError

class EmotionCommunicator:
    ip = ""
    port = 6666

    def __init__(self, ip="", port=6666):
        self.ip = ip
        self.port = port

    def send_image_and_emotions(self, open_socket, photo, emotions):
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


    def receive_image_and_emotions(self, open_socket):
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
    
    def print_connect(self, address=None):
        if address is None:
            print("Connected")
        else:
            print(f"Connected with: {address}")
        
    def print_disconnect(self, address=None):
        if address is None:
            print("Disconnected")
        else:
            print(f"Disconnected from: {address}")