# -*- coding:utf-8 -*-
# author: bgl
# weixin dingyuehao: Learn X
# date: 2019/7/1

import argparse

import cv2
import numpy as np


def draw_polylines(im, pts):
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(im, [pts], True, (0, 255, 255))
    return im


def vis(image, label):
    image = cv2.imread(image)
    labels = open(label, encoding='utf8').read().split('\n')
    for label in labels:
        label = np.array(label.split()[:8], dtype=np.float32).reshape([-1, 2]) * np.array(image.shape[:2][::-1])
        image = draw_polylines(image, label)

    cv2.imwrite('vis.jpg', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help='image path')
    parser.add_argument("--label", required=True, help='label path')
    args = parser.parse_args()

    vis(args.image, args.label)
