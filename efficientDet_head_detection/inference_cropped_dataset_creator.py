import cv2
import json
import numpy as np
import os
import time
import glob
import shutil
from pathlib import Path

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes


def main():
    debug = False
    out_dataset_path = "/home/tobi/cvcs/cropped_dataset/"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    phi = 0
    weighted_bifpn = True
    model_path = '/home/tobi/pascal_08_0.2204_0.3459.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]   
    # coco classes
    classes = {
    0:'head'
    }
    num_classes = 1
    score_threshold = 0.5
    color = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bifpn,
                            num_classes=num_classes,
                            score_threshold=score_threshold)
    model.load_weights(model_path, by_name=True)

    if not debug:
        if not os.path.exists(out_dataset_path):
            os.makedirs(out_dataset_path)
            os.makedirs(out_dataset_path + 'train/')
            os.makedirs(out_dataset_path + 'validation/')
            os.makedirs(out_dataset_path + 'test/')

    for image_path in glob.glob('/home/tobi/cvcs/cvcs_dataset/train/*.jpg'):
        
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)

        if scores[0] > score_threshold:
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
            box = boxes[0]
            label = labels[0]
            score = scores[0]

            new_im = crop_image(src_image, box)

            if debug:
                draw_boxes(src_image, box, score, label, color, classes)
                print(src_image.shape)

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('a', src_image)
                cv2.waitKey(0)

                cv2.namedWindow('image_new', cv2.WINDOW_NORMAL)
                cv2.imshow('b', new_im)
                cv2.waitKey(0)
            
            else:
                imname = str(Path(image_path).stem)
                cv2.imwrite(os.path.join(out_dataset_path, 'train', (imname + '.jpg')), new_im)
                shutil.copyfile(str(image_path).split('.')[0]+'.txt', os.path.join(out_dataset_path, 'train', (imname + '.txt')))
    
    for image_path in glob.glob('/home/tobi/cvcs/cvcs_dataset/validation/*.jpg'):
        
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)

        if scores[0] > score_threshold:
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
            box = boxes[0]
            label = labels[0]
            score = scores[0]

            new_im = crop_image(src_image, box)

            if debug:
                draw_boxes(src_image, box, score, label, color, classes)
                print(src_image.shape)

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('a', src_image)
                cv2.waitKey(0)

                cv2.namedWindow('image_new', cv2.WINDOW_NORMAL)
                cv2.imshow('b', new_im)
                cv2.waitKey(0)
            
            else:
                imname = str(Path(image_path).stem)
                cv2.imwrite(os.path.join(out_dataset_path, 'validation', (imname + '.jpg')), new_im)
                shutil.copyfile(str(image_path).split('.')[0]+'.txt', os.path.join(out_dataset_path, 'validation', (imname + '.txt')))
    
    for image_path in glob.glob('/home/tobi/cvcs/cvcs_dataset/test/*.jpg'):
        
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)

        if scores[0] > score_threshold:
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
            box = boxes[0]
            label = labels[0]
            score = scores[0]

            new_im = crop_image(src_image, box)

            if debug:
                draw_boxes(src_image, box, score, label, color, classes)
                print(src_image.shape)

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('a', src_image)
                cv2.waitKey(0)

                cv2.namedWindow('image_new', cv2.WINDOW_NORMAL)
                cv2.imshow('b', new_im)
                cv2.waitKey(0)
            
            else:
                imname = str(Path(image_path).stem)
                cv2.imwrite(os.path.join(out_dataset_path, 'test', (imname + '.jpg')), new_im)
                shutil.copyfile(str(image_path).split('.')[0]+'.txt', os.path.join(out_dataset_path, 'test', (imname + '.txt')))

    for image_path in glob.glob('/home/tobi/cvcs/cvcs_dataset/train/*.png'):
        
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)

        if scores[0] > score_threshold:
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
            box = boxes[0]
            label = labels[0]
            score = scores[0]

            new_im = crop_image(src_image, box)

            if debug:
                draw_boxes(src_image, box, score, label, color, classes)
                print(src_image.shape)

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('a', src_image)
                cv2.waitKey(0)

                cv2.namedWindow('image_new', cv2.WINDOW_NORMAL)
                cv2.imshow('b', new_im)
                cv2.waitKey(0)
            
            else:
                imname = str(Path(image_path).stem)
                cv2.imwrite(os.path.join(out_dataset_path, 'train', (imname + '.png')), new_im)
                shutil.copyfile(str(image_path).split('.')[0]+'.txt', os.path.join(out_dataset_path, 'train', (imname + '.txt')))

    for image_path in glob.glob('/home/tobi/cvcs/cvcs_dataset/validation/*.png'):
        
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)

        if scores[0] > score_threshold:
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
            box = boxes[0]
            label = labels[0]
            score = scores[0]

            new_im = crop_image(src_image, box)

            if debug:
                draw_boxes(src_image, box, score, label, color, classes)
                print(src_image.shape)

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('a', src_image)
                cv2.waitKey(0)

                cv2.namedWindow('image_new', cv2.WINDOW_NORMAL)
                cv2.imshow('b', new_im)
                cv2.waitKey(0)
            
            else:
                imname = str(Path(image_path).stem)
                cv2.imwrite(os.path.join(out_dataset_path, 'validaiton', (imname + '.png')), new_im)
                shutil.copyfile(str(image_path).split('.')[0]+'.txt', os.path.join(out_dataset_path, 'validation', (imname + '.txt')))

    for image_path in glob.glob('/home/tobi/cvcs/cvcs_dataset/test/*.png'):
        
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)

        if scores[0] > score_threshold:
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
            box = boxes[0]
            label = labels[0]
            score = scores[0]

            new_im = crop_image(src_image, box)

            if debug:
                draw_boxes(src_image, box, score, label, color, classes)
                print(src_image.shape)

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('a', src_image)
                cv2.waitKey(0)

                cv2.namedWindow('image_new', cv2.WINDOW_NORMAL)
                cv2.imshow('b', new_im)
                cv2.waitKey(0)
            
            else:
                imname = str(Path(image_path).stem)
                cv2.imwrite(os.path.join(out_dataset_path, 'test', (imname + '.png')), new_im)
                shutil.copyfile(str(image_path).split('.')[0]+'.txt', os.path.join(out_dataset_path, 'test', (imname + '.txt')))
        



def crop_image(src_image, box):
    xmin, ymin, xmax, ymax = list(map(int, box))
    out_im = src_image.copy()
    out_im = out_im[ymin:ymax+1, xmin:xmax+1, :]
    print(out_im.shape)
    return out_im

if __name__ == '__main__':
    main()
