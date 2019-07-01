# -*- coding:utf-8 -*-
# author: bgl
# weixin dingyuehao: Learn X
# date: 2019/7/1

import argparse
import glob
import os
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def binarize(im):
    ret, dst = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return dst


def draw_polylines(im, pts):
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(im, [pts], True, (0, 255, 255))
    return im


def blur(img, sigma=1, kernel=3):
    return cv2.GaussianBlur(img, (kernel, kernel), sigma)


def warp_affine_point(points, perspective_mat):
    points = np.float32(points).reshape([-1, 2])
    out_point = np.matmul(points, perspective_mat[:, :2].T) + perspective_mat[:, 2].T
    out_point = np.array(out_point, dtype=np.float32).reshape([-1, 2])
    return out_point


def rotate_no_loss(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    in_points = [0, 0, w, 0, w, h, 0, h]
    out_points = warp_affine_point(in_points, M)
    out_points = np.reshape(np.array(out_points, dtype=np.float32), [4, 2])
    minx, miny = np.min(out_points, axis=0)
    w, h = np.max(out_points, axis=0) - np.min(out_points, axis=0)
    out_points -= np.array([minx, miny])

    src = np.reshape(np.float32(in_points), [4, 2])
    dst = np.reshape(np.float32(out_points), [4, 2])
    M = cv2.getPerspectiveTransform(src, dst)
    rotated = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return rotated, angle, out_points


def get_text_bmp(text, font, font_size, color):
    font = ImageFont.truetype(font, size=font_size, encoding="utf-8")
    size = font.getsize(text)

    im = np.zeros([size[1], size[0], 4], dtype=np.uint8)
    im[..., 3] = 0
    pilim = Image.fromarray(im)
    draw = ImageDraw.Draw(pilim)

    pos = [0, 0]
    draw.text(pos, text, fill=color, font=font)
    im = np.array(pilim)
    return im


def put_bmp(im, bmp, pos, size):
    bmp = cv2.resize(bmp, tuple(size), cv2.INTER_AREA)
    bmp[..., 3] = binarize(bmp[..., 3])
    target = im[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]]
    for i in range(3):
        target[..., i] = np.where(bmp[..., 3] < 50, target[..., i], bmp[..., i])
    im[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]] = target
    return im


def generate(background_dir, out_dir, count, show=False):
    path_list = glob.glob(os.path.join(background_dir, '*.jpg'))
    path_list += glob.glob(os.path.join(background_dir, '*.png'))
    image_dir = os.path.join(out_dir, 'image')
    label_dir = os.path.join(out_dir, 'label')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for i in range(count):
        path = path_list[np.random.randint(len(path_list))]

        image = cv2.imread(path)
        h, w = image.shape[:2]
        x = np.random.randint(w // 2)
        y = np.random.randint(h // 2)
        x1 = np.random.randint(w // 2, w)
        y1 = np.random.randint(h // 2, h)
        image = image[y:y1, x:x1]

        image, box_list = put_texts(image)

        name = str(uuid.uuid1())
        label_file = os.path.join(out_dir, 'label', name + '.txt')
        label_file = open(label_file, mode='w', encoding='utf8')
        for box in box_list:
            if show:
                image = draw_polylines(image, box)
            box = np.array(box, dtype=np.float32).reshape([-1, 2]) / np.array(image.shape[:2][::-1])
            box = box.reshape([-1])
            label_file.write(' '.join([str(b) for b in box]) + '\n')

        image_file = os.path.join(out_dir, 'image', name + '.jpg')
        cv2.imwrite(image_file, image)
        if i % 100 == 0:
            print('processed: ', i)


def put_texts(im):
    atoz = ''.join([chr(c) for c in range(ord('a'), ord('z'))])
    AtoZ = ''.join([chr(c) for c in range(ord('A'), ord('Z'))])
    atozAtoZ = atoz + AtoZ
    line_num = 11
    w = 520
    h = 400
    font_list = ['LiberationMono-Bold.ttf', 'NotoMono-Regular.ttf', 'LiberationSans-Bold.ttf', 'Sawasdee-Bold.ttf',
                 'FreeSansBold.ttf']
    im = cv2.resize(im, (w, h))
    line_height = im.shape[0] // line_num
    min_word_per_line = 0
    max_word_per_line = 9
    min_space_width = 20
    max_space_width = 30
    min_font_size = 18
    max_font_size = 25
    min_right_padding = 10
    min_char_per_word = 2
    max_char_per_word = 9

    box_list = []
    for i in range(line_num):
        y = i * line_height
        x = np.random.randint(h // 2)
        for j in range(np.random.randint(min_word_per_line, max_word_per_line)):
            xi = x + np.random.randint(min_space_width, max_space_width)
            word = ''.join([atozAtoZ[np.random.randint(len(atozAtoZ))] for _ in
                            range(np.random.randint(min_char_per_word, max_char_per_word))])
            font_size = np.random.randint(min_font_size, max_font_size)
            font = font_list[np.random.randint(len(font_list))]

            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255),
                     np.random.randint(128, 255))
            yi = y + np.random.randint(line_height // 10)

            bmp = get_text_bmp(word, font, font_size, color)
            bmp, angle, out_points = rotate_no_loss(bmp, np.random.randint(-10, 10))
            s = bmp.shape
            if s[1] + xi > w - min_right_padding:
                continue
            if s[0] + yi > h - min_right_padding:
                continue
            size = (s[1], s[0])
            im = put_bmp(im, bmp, [xi, yi], size)
            four_point = np.reshape(np.array(out_points), [4, 2]) + np.array([xi, yi])

            x = xi + size[0]
            box = four_point.reshape([-1]).tolist()
            box_list.append(box)
    im = blur(im)

    return im, box_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--back_dir", required=True, help='background image directory')
    parser.add_argument("--out_dir", required=True, help='output result directory')
    args = parser.parse_args()
    count = args.count
    show = False

    generate(args.back_dir, args.out_dir, args.count, show=show)
