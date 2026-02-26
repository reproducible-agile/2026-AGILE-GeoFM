import argparse
import cv2
import numpy as np
import os
import random
import shapely
from scipy.ndimage import maximum_filter

from PIL import Image
from rasterio import features


def saveclip(images, out_dir, img_name):
    for i, img in enumerate(images):
        save_path = os.path.join(out_dir, f'{img_name}_{i}' + '.png')
        img.save(save_path)


def mask_to_prompt(mask_array):
    mask = mask_array == 0
    shapes = features.shapes(mask_array, mask)
    polygons = [shapely.geometry.shape(shape) for shape, _ in shapes]

    if polygons:
        bb = shapely.envelope(polygons)
        return [list(box.bounds) for box in bb]
    else:
        return []


def main(args):
    dataset = args.dataset
    img_dir = f'data/{dataset}/'
    out_dir = f'data/{dataset}_cache/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    images = []

    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    img_name = random.choice(files)
    image = Image.open(os.path.join(img_dir, img_name))
    image = np.array(image)

    gt_mask = cv2.imread(os.path.join(img_dir, img_name[:-9] + 'osm.png'), 0).astype(np.uint8)
    prompts = mask_to_prompt(gt_mask)
    gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY_INV)[1]

    mask_buffered = maximum_filter(gt_mask, size=11, mode='constant', cval=0)>0

    masked = image * np.expand_dims(mask_buffered, -1)
    for prompt in prompts:
        crop = masked[max(int(prompt[1])-5, 0) : min(int(prompt[3])+5, 1024), max(int(prompt[0])-5,0) : min(int(prompt[2])+5, 1024)]
        h = int(prompt[3] - prompt[1])
        w = int(prompt[2] - prompt[0])

        if h > w:
            top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
        else:
            top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
        # padding
        crop = cv2.copyMakeBorder(
            crop,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        crop = Image.fromarray(crop)
        images.append(crop)

    saveclip(images, out_dir, img_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbd1k', choices=['bbd1k', 'water1k'])
    args = parser.parse_args()

    main(args)
