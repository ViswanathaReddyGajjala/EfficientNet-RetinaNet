import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BBoxTransform(nn.Module):
    """[summary]"""

    def __init__(self, mean=None, std=None):
        """[summary]

        Args:
            mean ([type], optional): [description]. Defaults to None.
            std ([type], optional): [description]. Defaults to None.
        """
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).to(
                device
            )
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(
                np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
            ).to(device)
        else:
            self.std = std

    def forward(self, boxes, deltas):
        """computes the output coordinates

        Args:
            boxes ([tensor]): [description]
            deltas ([tensor]): [description]

        Returns:
            [tensor]: [description]
        """

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2
        )

        return pred_boxes


class ClipBoxes(nn.Module):
    """[summary]"""

    def __init__(self):
        """[summary]"""
        super(ClipBoxes, self).__init__()

    @staticmethod
    def forward(boxes, img):
        """[clips the box coordinates]

        Args:
            boxes ([tensor]): [bounding box coordinates]
            img ([tensor]): [image tensor]

        Returns:
            [type]: [description]
        """
        # batch_size, num_channels, height, width = img.shape
        _, _, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes
