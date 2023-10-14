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