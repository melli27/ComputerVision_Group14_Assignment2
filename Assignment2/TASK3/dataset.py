# TUWIEN - CV: Task3 - Scene recognition using Bag of Visual Words
# *********+++++++++*******++++ GROUP NO. 14 ++++*******+++++++++*********
import glob
import os
import cv2
from typing import List, Tuple
import numpy as np

class SceneDataset:
    images: List[np.ndarray] = []         # list of images
    labels: List[int] = []                # list of labels of images
    class_names: List[str] = []           # list with of class names (folder names)

    def __init__(self, path: str) -> None:
        """
        Initializes SceneDataset object and processes images and labels from the given path.

        Args:
        - path (str): Path to the dataset folder.
        """
        img_data = []
        labels = []
        dirs = []

        # Loop through all subfolders within the given 'path', get all images per folder,
        # save the images in gray scale and normalize the image values between 0 and 1.
        # The label of an image is the current subfolder (e.g., value between 0-9 when using 10 classes).
        # HINT: os.listdir(..), glob.glob(..), cv2.imread(..)
        # student_code start

        # list subfolder such as bedroom, coast, forest, ...
        subfolders = os.listdir(path)
        # Sort subfolder, so the idx for the labels are the same for test and train data
        subfolders = sorted(subfolders)

        for label_idx, folder_name in enumerate(subfolders):
            # Save class names
            dirs.append(folder_name)

            # Create subfolder path
            folder_path = os.path.join(path, folder_name, "*.jpg")
            # Find all images in subfolder
            image_paths = glob.glob(folder_path)

            for img_path in image_paths:
                # Save image as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # Resize image to 100 x 100 only for OWN test images, because it cv2 is not importet in notebooke area
                img = cv2.resize(img, (100, 100))
                # Normalize values between 0 and 1
                img = img / 255.0

                img_data.append(img)
                labels.append(label_idx)   

        # student_code end

        # Save as local parameters
        self.images = img_data
        self.labels = labels
        self.class_names = dirs

    def get_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        Returns images and their corresponding labels.

        Returns:
        - Tuple containing a list of images and a list of labels.
        """
        return self.images, self.labels

    def get_class_names(self) -> List[str]:
        """
        Returns the list of class names.

        Returns:
        - List of class names.
        """
        return self.class_names