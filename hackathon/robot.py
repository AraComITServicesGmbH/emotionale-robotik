from queue import Queue
import time


class Robot:
    update_interval = 0.5 
    
    def update(self,  image, emotions):
        """
        Updates the current instance with a new image and associated emotions.

        Parameters:
        - image (numpy.ndarray): The BGR image to be processed.
        - emotions (list): A list containing the emotions associated with the image.
        """
        raise NotImplementedError()
    
    def update_from_queue(self, queue: Queue):
        """
        Continuously update the robot with photos and corresponding emotions 
        from the queue at the robot's specified update interval. After processing 
        an item from the queue, it clears the rest of the queue to ensure the 
        robot only receives the latest information.

        Parameters:
        - queue (Queue): A queue containing tuples of photos and corresponding emotions.

        Note:
        This function runs in an infinite loop and should typically be executed in a separate thread.
        """
        while True:
            photo, emotions = queue.get()
            with queue.mutex:
                queue.queue.clear()
            self.update(photo, emotions)
            time.sleep(self.update_interval)

    def print_emotions(self, emotions):
        print(f"{len(emotions)} Persons with the following emotions: ")
        print(" ".join(emotions))