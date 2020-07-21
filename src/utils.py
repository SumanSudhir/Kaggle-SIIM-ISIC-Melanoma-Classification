import cv2
import random
import numpy as np

class DrawHair:
    """
    Draw a random number of pseduo hairs

    Args:
    hairs (int): maximum number of hairs to draw
    width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width: tuple = (1,2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL IMage: Image with drwan hairs
        """
        if not self.hairs:
            return img

        width, height = img.size
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0,width), random.randint(0,height//2))
            # The end of the line
            end = (random.randint(0, width), random.randint(0, height))
            # Black color of the hair
            color = (0, 0, 0)
            cv2.line(np.array(img), origin, end, color, random.randint(self.width[0], self.width[1]))

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'
