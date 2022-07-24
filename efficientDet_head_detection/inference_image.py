import cv2
import json
import numpy as np
import os
import time
import glob
from pathlib import Path

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score, recall_score
import argparse
import os
import cv2


#import sys, os
#sys.path.insert(0, os.path.abspath('..'))
#from xception_detection.models import model_selection
#from xception_detection.dataset.transform import xception_default_data_transforms
#from xception_detection.dataset.mydataset import MyDataset



def main():
	debug = False
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


	####XCEPTION
	#test_dataset = MyDataset(txt_path='/home/tobi/cvcs/official/CVCS_project_DeepFake/inference_list.txt', transform=xception_default_data_transforms['test'])
	#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=1)
	
	#deepFakeDetector
	#model = model_selection(modelname='midwayxception', num_out_classes=2, dropout=0.5)
	#model.load_state_dict(torch.load(model_path))


	num_classes = 1
	score_threshold = 0.5
	color = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
	_, model = efficientdet(phi=phi,
							weighted_bifpn=weighted_bifpn,
							num_classes=num_classes,
							score_threshold=score_threshold)
	model.load_weights(model_path, by_name=True)

	for image_path in glob.glob('/home/tobi/cvcs/fe/resize/*.jpg'):
		
		image = cv2.imread(image_path)

		#image = cv2.resize(image, (int(image.shape[0]*50/100), int(image.shape[1]*50/100)))
		
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

			#new_im = crop_image(src_image, box)
			purple = (153,0,153)
			green = (0,255,0)
			draw_boxes(src_image, box, score, label, green, classes)

			if debug:
				draw_boxes(src_image, box, score, label, color, classes)
				print(src_image.shape)

				cv2.namedWindow('image', cv2.WINDOW_NORMAL)
				cv2.imshow('a', src_image)
				cv2.waitKey(0)

				#cv2.namedWindow('image_new', cv2.WINDOW_NORMAL)
				#cv2.imshow('b', new_im)
				#cv2.waitKey(0)
			
			else:
				imname = str(Path(image_path).stem)
				cv2.imwrite('/home/tobi/out_pred_imgs/real/' + str(imname) + '.jpg', src_image)

def crop_image(src_image, box):
	xmin, ymin, xmax, ymax = list(map(int, box))
	out_im = src_image.copy()
	out_im = out_im[ymin:ymax+1, xmin:xmax+1, :]
	print(out_im.shape)
	return out_im

if __name__ == '__main__':
	main()
