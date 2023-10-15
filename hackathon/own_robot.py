from robot import Robot
import cv2 as cv

import os


class OwnRobot(Robot):
    count = 0

    def update(self, image, emotions):
        self.count += 1
        if self.count > 10:
            self.count = 1
        super().print_emotions(emotions)
        cv.imwrite(os.sep.join(["img", f"ownRobot{self.count}.png"]), image)
