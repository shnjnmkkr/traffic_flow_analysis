import numpy as np
import cv2

class GMMBackgroundSubtractor:
    """
    Gaussian Mixture Model (GMM) for background subtraction and foreground extraction.
    Uses OpenCV's BackgroundSubtractorMOG2 for simplicity and speed.
    """
    def __init__(self, history=500, varThreshold=16, detectShadows=True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=varThreshold,
            detectShadows=detectShadows
        )

    def apply(self, frame):
        """
        Apply GMM to a frame and return the foreground mask.
        Args:
            frame (np.ndarray): Input BGR image.
        Returns:
            fg_mask (np.ndarray): Foreground mask (uint8, 0 or 255).
        """
        fg_mask = self.bg_subtractor.apply(frame)
        # Optional: clean up mask
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        return fg_mask

# Example usage:
# gmm = GMMBackgroundSubtractor()
# fg_mask = gmm.apply(frame) 