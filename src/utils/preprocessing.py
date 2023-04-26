import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import io
from math import ceil, floor

# import seaborn as sns
from tqdm import tqdm
from torchvision import transforms

# implement this class for preprocessor
# or make edit in this class and remove @abstractmethod decorator
class Preprocessor:
    """
    Class with functions to preprocess images on the fly.
    Required Functions:
    - get:
        - Input:
            - image_path: str
                - path to image
            - transforms: list(str/int/other format)
                - list of transforms such as rotate/flip/resize/etc.
        - Returns:
            - image: np.ndarray (w, h, 3)
                - image matrix
    """

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def make_random_combinations(
        self,
        num_cominations,
        p_transformations={
            "rotate": 0.5,
            "scale": 0.5,
            "flip": 0.5,
            "gaussian_blur": 0.5,
            "color_jitter": 0.5,
            "random_erasing": 0.5,
        },
    ):
        """
        Generates random combinations of transformations to be applied to the images.
            - p_transformations: dictionary of probabilities of each transformation
        """

        combinations = []
        for i in range(num_cominations):
            combination = ""

            if (
                "rotate" in p_transformations
                and np.random.random() < p_transformations["rotate"]
            ):
                combination += "r"
            if (
                "scale" in p_transformations
                and np.random.random() < p_transformations["scale"]
            ):
                combination += "s"
            if (
                "flip" in p_transformations
                and np.random.random() < p_transformations["flip"]
            ):
                combination += "f"
            if (
                "gaussian_blur" in p_transformations
                and np.random.random() < p_transformations["gaussian_blur"]
            ):
                combination += "g"
            if (
                "color_jitter" in p_transformations
                and np.random.random() < p_transformations["color_jitter"]
            ):
                combination += "c"
            if (
                "random_erasing" in p_transformations
                and np.random.random() < p_transformations["random_erasing"]
            ):
                combination += "e"

            combinations.append(combination)
        return combinations

    # perform data augmentation on the images
    def augment(
        self,
        image,
        combination=None,
        rotate=None,
        scale=None,
        flip=None,
        together=False,
        gaussian_blur=None,
        color_jitter=None,
        random_erasing=None,
    ):
        """
        image: tensor of shape (C, H, W)
        Rotate, scale and flip the image
        Rotate: Angle of rotation
        Scale: Scale factor
        Flip: 'h' for horizontal flip, 'v' for vertical flip
        Combination: 'r' for rotate, 's' for scale, 'f' for flip, 'rs' for rotate and scale, 'rf' for rotate and flip, 'sf' for scale and flip, 'rsf' for rotate, scale and flip
        Together: True if all the transformations are to be applied together, False if they are to be applied separately
        """
        if combination is not None:
            if together:
                for c in combination:
                    # can make it more readable if necessary using complete names and a list.
                    if c == "r":
                        image = self.augment(image, rotate=rotate)
                    elif c == "s":
                        image = self.augment(image, scale=scale)
                    elif c == "f":
                        image = self.augment(image, flip=flip)
                    elif c == "g":
                        image = self.augment(image, gaussian_blur=gaussian_blur)
                    elif c == "c":
                        image = self.augment(image, color_jitter=color_jitter)
                    elif c == "e":
                        image = self.augment(image, random_erasing=random_erasing)

                return image

        if rotate is not None:
            image = transforms.functional.rotate(image, rotate)

        if scale is not None:
            cropped_image = transforms.functional.center_crop(
                image, image.shape[1] * scale
            )
            # pad the cropped image to the original size
            pad = image.shape[1] - cropped_image.shape[1]
            image = transforms.functional.pad(
                cropped_image,
                (ceil(pad / 2), ceil(pad / 2), floor(pad / 2), floor(pad / 2)),
            )

        if flip is not None:
            if flip == "h":
                image = torch.flip(image, [2])
            elif flip == "v":
                image = torch.flip(image, [1])

        if gaussian_blur is not None:
            for _ in range(gaussian_blur):
                image = transforms.functional.gaussian_blur(
                    image, kernel_size=(5, 9), sigma=(0.1, 5)
                )

        if color_jitter is not None:
            image = transforms.functional.adjust_brightness(
                image, color_jitter["brightness"]
            )
            image = transforms.functional.adjust_contrast(
                image, color_jitter["contrast"]
            )
            image = transforms.functional.adjust_saturation(
                image, color_jitter["saturation"]
            )
            image = transforms.functional.adjust_hue(image, color_jitter["hue"])

        if random_erasing is not None:
            image = transforms.RandomErasing(p=random_erasing, inplace=False)(image)

        return image

    def get(
        self,
        combination="",
        # image_path="",
        image_data = np.zeros(3*32*32),
        rotate=np.random.randint(0, 90),
        scale=np.random.uniform(0.5, 1),
        flip=np.random.choice(["h", "v"]),
        gaussian_blur=np.random.randint(1, 6),
        color_jitter={
            "brightness": np.random.uniform(0.5, 1.5),
            "contrast": np.random.uniform(0.5, 1.5),
            "saturation": np.random.uniform(0.5, 1.5),
            "hue": np.random.uniform(-0.5, 0.5),
        },
        random_erasing=None,
        together=True,
    ):

        # image = Image.open(image_path)
        # image = image.resize(self.image_size)
        # image = np.array(image)
        image = torch.from_numpy(image_data)
        # image = image.permute(2, 0, 1)
        image = image.float()
        image = image / 255
        # image = self.augment(
        #     image,
        #     combination=combination,
        # )
        image = self.augment(
            image,
            combination=combination,
            rotate=rotate,
            scale=scale,
            flip=flip,
            gaussian_blur=gaussian_blur,
            color_jitter=color_jitter,
            random_erasing=random_erasing,
            together=together,
        )

        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()
        return image
