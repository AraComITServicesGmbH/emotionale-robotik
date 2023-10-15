from queue import Queue
import socket
from threading import Thread
import time
from emotion_communicator import EmotionCommunicator


class EmotionServer(EmotionCommunicator):
    def __init__(self, ip="", port=6666, update_rate=2):
        super().__init__(ip, port)
        self.update_rate = update_rate

    def create_server_socket(self):
        """
        Creates a server socket bound to the specified IP address and port.

        This function initializes a new server socket using the IPv4 address family
        and the TCP protocol. It then binds the socket to the given IP address and port.
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((super().ip, super().port))

    def update_to_clients(self, queue: Queue):
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
        self.create_server_socket()
        try:
            self.server_socket.listen(5)
            while True:
                client, address = self.server_socket.accept()
                thread = Thread(
                    target=self.update_to_client, args=(client, address, queue)
                )
                thread.start()
        finally:
            self.server_socket.close()


    def update_to_client(self, client, address, queue):
        """
        Sends updates to a client at a specified rate defined by update rate 
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
            super().print_connect(address)
            while True:
                time_elapsed = time.time() - previous_time
                if time_elapsed > 1/ self.update_rate:
                    time_elapsed = time.time()
                    (frame_bgr, emotions) = queue.get()
                    super().send_image_and_emotions(client, frame_bgr, emotions)
        except Exception:
            super().print_disconnect(address)
            client.close()