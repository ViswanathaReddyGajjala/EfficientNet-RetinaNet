#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:49:21 2019

@author: viswanatha
"""

from __future__ import print_function, division
import torch
import numpy as np
import skimage.io
import skimage.transform
import skimage.color
import skimage
from dataloader import UnNormalizer

import time
import cv2
import argparse


class Resize_Img:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, min_side=608, max_side=1024):
        image = img

        rows, cols, cns = image.shape

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

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return {"img": torch.from_numpy(new_image), "scale": scale}


def collat(data):
    imgs = data
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = 1

    # print (batch_size)
    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        img = torch.Tensor(img)
        padded_imgs[i, : int(img.shape[0]), : int(img.shape[1]), :] = img

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return padded_imgs


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
    )


def visualize(args):
    model_path = args.model_path
    image_path = args.image_path
    use_gpu = args.use_gpu
    retinanet = torch.load(model_path)

    custom_labels = {"cobia"}
    label_map = {k: v + 1 for v, k in enumerate(custom_labels)}
    label_map["background"] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    if use_gpu:
        retinanet = retinanet.cuda()

    unnormalize = UnNormalizer()
    retinanet.eval()

    with torch.no_grad():
        st = time.time()
        img = cv2.imread(image_path)
        img = img.astype(np.float32) / 255.0

        mean = np.array([[[0.485, 0.456, 0.406]]])
        std = np.array([[[0.229, 0.224, 0.225]]])
        img = (img.astype(np.float32) - mean) / std

        image_resizer = Resize_Img()
        img = image_resizer(img)["img"]

        img = np.expand_dims(img, axis=0)
        img = collat(img)
        scores, classification, transformed_anchors = retinanet(img.cuda().float())
        print("Elapsed time: {}".format(time.time() - st))
        idxs = np.where(scores.cpu() > 0.5)
        img = np.array(255 * unnormalize(img[0, :, :, :])).copy()

        img[img < 0] = 0
        img[img > 255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = rev_label_map[int(classification[idxs[0][j]])]
            draw_caption(img, (x1, y1, x2, y2), label_name)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print(label_name, x1, y1, x2, y2)

        cv2.imwrite("out.png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("image_path", help="Path to test image")
    parser.add_argument("use_gpu", help="True if gpu is available")
    arguments = parser.parse_args()

    visualize(arguments)
