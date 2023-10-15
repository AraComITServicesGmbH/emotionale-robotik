from queue import Queue
import socket
import time

from emotion_communicator import EmotionCommunicator


class EmotionClient(EmotionCommunicator):
    def create_client_socket(self):
        """
        Initialize and configure a TCP client socket using the IPv4 address family.
        
        This method sets up a new client socket for communication over TCP, 
        configuring the socket's send buffer size to 100,000 bytes to enhance 
        sending performance.
        """
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 100000)

    def update_from_server(self, queue: Queue):
        """
        Persistently attempt to establish a connection to the server and 
        populate the provided queue with image and emotion data.
        
        In the event of a connection loss, it will attempt to re-establish 
        the connection.

        Parameters:
        - queue (Queue): A queue in which the received image and emotion data 
          will be stored.

        Note:
        This function runs indefinitely. Typically, it should be invoked within 
        a separate thread to avoid blocking the main program.
        """
        while True:
            self.create_client_socket()
            try:
                self.client_socket.connect((self.ip, self.port))
                super().print_connect()
                while True:
                    result = super().receive_image_and_emotions(self.client_socket)
                    if result is not None:
                        queue.put(result)
            except Exception: # Expected exceptions include disconnects or EOFError.
                super().print_disconnect()
                self.client_socket.close()
            time.sleep(1)
