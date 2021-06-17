from __future__ import print_function, division
import sys
import torch
import numpy as np
import random
import csv
import cv2

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=","))
        except ValueError as e:
            raise ValueError(
                "invalid CSV class file: {}: {}".format(self.class_list, e)
            )

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(
                    csv.reader(file, delimiter=","), self.classes
                )
        except ValueError as e:
            raise ValueError(
                "invalid CSV annotations file: {}: {}".format(self.train_file, e)
            )
        self.image_names = list(self.image_data.keys())

    @staticmethod
    def _parse(value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))

    @staticmethod
    def _open_for_csv(path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, "rb")
        return open(path, "r", newline="")

    def load_classes(self, csv_reader):
        """[summary]

        Args:
            csv_reader ([type]): [description]

        Raises:
            ValueError: [if the annotation format is wrong]

        Returns:
            [dict]: [contains clas_name and the respective id]
        """
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise ValueError(
                    "line {}: format should be 'class_name,class_id'".format(line)
                )
            class_id = self._parse(
                class_id, int, "line {}: malformed class ID: {{}}".format(line)
            )

            if class_name in result:
                raise ValueError(
                    "line {}: duplicate class name: '{}'".format(line, class_name)
                )
            result[class_name] = class_id
        return result

    def __len__(self):
        """[summary]

        Returns:
            [int]: [total number of images]
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([int]): [retrieves the image at the respective index]

        Returns:
            [dict]: [{'img': img, 'annot': annot}]
        """

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {"img": img, "annot": annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        """[summary]

        Args:
            image_index ([int]): [retrieves the image at the respective index]

        Returns:
            [numpy array]: [image]
        """
        # img = skimage.io.imread(self.image_names[image_index])

        img = cv2.imread(self.image_names[image_index])
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        """[summary]

        Args:
            image_index ([int]): [retrieves the image annotations
                                    at the respective index]

        Returns:
            [list]: [image annotations]
        """
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for _, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a["x1"]
            x2 = a["x2"]
            y1 = a["y1"]
            y2 = a["y2"]

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a["class"])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        """[summary]

        Args:
            csv_reader ([type]): [description]
            classes ([type]): [description]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise ValueError(
                    "line {}: format should be 'img_file,x1,y1,x2,y2,class_name' or 'img_file,,,,,'".format(
                        line
                    )
                )

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ("", "", "", "", ""):
                continue

            x1 = self._parse(x1, float, "line {}: malformed x1: {{}}".format(line))
            y1 = self._parse(y1, float, "line {}: malformed y1: {{}}".format(line))
            x2 = self._parse(x2, float, "line {}: malformed x2: {{}}".format(line))
            y2 = self._parse(y2, float, "line {}: malformed y2: {{}}".format(line))

            # Check that the bounding box is valid.
            # if x2 <= x1:
            #    raise ValueError('line {}: x2 ({}) must be higher
            #                       than x1 ({})'.format(line, x2, x1))
            # if y2 <= y1:
            #    raise ValueError('line {}: y2 ({}) must be higher
            #                       than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError(
                    "line {}: unknown class name: '{}' (classes: {})".format(
                        line, class_name, classes
                    )
                )

            result[img_file].append(
                {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "class": class_name}
            )
        return result

    def name_to_label(self, name):
        """maps class name to id

        Args:
            name ([str]): [class name]

        Returns:
            [int]: [class id]
        """
        return self.classes[name]

    def label_to_name(self, label):
        """[maps class id to class name]

        Args:
            label ([int]): [class id]

        Returns:
            [str]: [class name]
        """
        return self.labels[label]

    def num_classes(self):
        """total number of classes + background

        Returns:
            [int]: [total number of class]
        """
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        """computes the aspect ratio for the image at the given index.

        Args:
            image_index ([int]): []

        Returns:
            [float]: [image aspect ratio]
        """
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    """[summary]

    Args:
        data ([dict]): [description]

    Returns:
        [dict]: [description]
    """
    imgs = [s["img"] for s in data]
    annots = [s["annot"] for s in data]
    scales = [s["scale"] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, : int(img.shape[0]), : int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, : annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {"img": padded_imgs, "annot": annot_padded, "scale": scales}


class Resizer(object):
    """[summary]"""

    def __init__(self, is_test=False, multi_scale=False, p=0.7):
        """[summary]

        Args:
            is_test (bool, optional): [train/test phase]. Defaults to False.
            multi_scale (bool, optional): [enable/disable multi scale during training]. Defaults to False.
            p (float, optional): [probability to use multi_scale]. Defaults to 0.7.
        """
        self.is_test = is_test
        self.p = p
        self.multi_scale = multi_scale

    def __call__(self, sample):
        """[summary]

        Args:
            sample ([dict]): [contains image and image annotations]

        Returns:
            [dict]: [contains resized image and image annotations.
                        scale that used for resizing]
        """
        image, annots = sample["img"], sample["annot"]
        min_side = 608
        max_side = 1024

        if self.multi_scale and (not self.is_test) and np.random.rand() < self.p:
            min_side = random.choice(
                [32 * 20, 32 * 21, 32 * 22, 32 * 23, 32 * 24, 32 * 25]
            )
            # max_side = random.choice([32*20, 32*21, 32*22, 32*23, 32*24, 32*25])
            max_side_scale = random.choice(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                ]
            )
            max_side = min_side + (32 * max_side_scale)

        rows, cols, cns = image.shape
        # min_size = random.sa

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(
            image, (int(round(rows * scale)), int(round((cols * scale))))
        )
        rows, cols, cns = image.shape

        pad_w, pad_h = 0, 0
        if rows % 32 != 0:
            pad_w = 32 - rows % 32
        if cols % 32 != 0:
            pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {
            "img": torch.from_numpy(new_image),
            "annot": torch.from_numpy(annots),
            "scale": scale,
        }


class Augmenter:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        """[horizontal flip]

        Args:
            sample ([dict]): [contains image and image annotations]
            flip_x (float, optional): [probability to flip an image]. Defaults to 0.5.

        Returns:
            [dict]: [contains transformed image and image annotations]
        """

        if np.random.rand() < flip_x:
            image, annots = sample["img"], sample["annot"]
            image = image[:, ::-1, :]

            _, cols, _ = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {"img": image, "annot": annots}

        return sample


class Normalizer:
    """[image normalization]"""

    def __init__(self):
        """[summary]"""
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        """[summary]

        Args:
            sample ([dict]): [contains image and image annotations]

        Returns:
            [dict]: [normalized image and image annotations]
        """
        image, annots = sample["img"], sample["annot"]

        return {
            "img": ((image.astype(np.float32) - self.mean) / self.std),
            "annot": annots,
        }


class UnNormalizer:
    """[summary]"""

    def __init__(self, mean=None, std=None):
        """[summary]

        Args:
            mean ([list], optional): [description]. Defaults to None.
            std ([list], optional): [description]. Defaults to None.
        """
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    """[summary]

    Args:
        Sampler ([type]): [description]
    """

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [
            [order[x % len(order)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(order), self.batch_size)
        ]
