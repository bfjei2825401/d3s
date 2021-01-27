# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np


def attach_mask_save(img_path, mask_img_path, masked_save_path, transparency=0.3, mask_color=(255, 140, 0), border=2):
    """
    attach the mask image to the origin image with an bold contour
    :param img_path: image file path
    :param mask_img_path: mask file path, file extension .png
    :param masked_save_path: mask attached image save path
    :param transparency: mask transparency
    :param mask_color: mask color
    :param border: mask border width, no border if is None
    :return:
    """
    if not os.path.exists(img_path):
        print("image file does not exists! " + img_path)
        return
        # raise Exception("image file does not exists!")
    if not os.path.exists(mask_img_path):
        print("mask image file does not exists! " + mask_img_path)
        return
        # raise Exception("mask image file does not exists! " + mask_img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_img_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_save = attach_mask(img, mask * 255, transparency, mask_color, border)
    img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
    save_dir = os.path.dirname(masked_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(masked_save_path, img_save)


def attach_mask(img, mask, transparency=0.3, mask_color=(255, 140, 0), border=2):
    """

    :param img: RGB numpy image
    :param mask: GRAY image
    :param transparency: mask transparency
    :param mask_color: mask color
    :param border: mask border width, no border if is None
    :return: masked image in numpy form
    """
    copy_img = img.copy()
    _, mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
    if cv2.__version__[0] == '4':
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if border:
        contoured = cv2.polylines(img, contours[0], True, mask_color, border)
    else:
        contoured = cv2.polylines(img, contours[0], True, mask_color, 0)
    filled = cv2.fillPoly(copy_img, contours, mask_color)
    beta = 1 - transparency
    masked_img = cv2.addWeighted(filled, transparency, contoured, beta, 0)
    return masked_img
