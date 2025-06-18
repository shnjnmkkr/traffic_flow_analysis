import numpy as np

class VehicleCounter:
    """
    Counts vehicles per frame by matching detection bounding boxes with GMM foreground masks.
    """
    def __init__(self, min_overlap=0.3):
        self.min_overlap = min_overlap  # Minimum IoU/overlap ratio to consider a detection as foreground

    def count(self, bboxes, fg_mask):
        """
        Count vehicles by checking if detection boxes overlap with foreground mask.
        Args:
            bboxes (list of [x1, y1, x2, y2]): Detected bounding boxes
            fg_mask (np.ndarray): Foreground mask (uint8, 0 or 255)
        Returns:
            count (int): Number of vehicles detected in foreground
        """
        count = 0
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            box_mask = fg_mask[y1:y2, x1:x2]
            if box_mask.size == 0:
                continue
            overlap = np.count_nonzero(box_mask) / float(box_mask.size)
            if overlap > self.min_overlap:
                count += 1
        return count

# Example usage:
# counter = VehicleCounter(min_overlap=0.3)
# count = counter.count(bboxes, fg_mask) 