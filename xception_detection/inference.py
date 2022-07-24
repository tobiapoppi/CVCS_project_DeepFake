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

from models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
#import efficientDet_head_detection

def main():
	args = parse.parse_args()
	model_path = args.model_path
	torch.backends.cudnn.benchmark=True

	test_dataset = MyDataset(txt_path=args.input_list, transform=xception_default_data_transforms['test'])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=1)
	
	#deepFakeDetector
	model = model_selection(modelname='midwayxception', num_out_classes=2, dropout=0.5)
	model.load_state_dict(torch.load(model_path))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	model.eval()
	

	#headDetector
	with torch.no_grad():
		for (i,l) in test_loader:
			i = i.cuda()
			output = model(i)
			_, preds = torch.max(output.data, 1)
			
			print('true: ',l)
			print('pred: ', preds)
			print('\n')




if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--input_list', '-il', type=str, default='/home/tobi/cvcs/official/CVCS_project_DeepFake/inference_list.txt')
	parse.add_argument('--model_path', '-mp', type=str, default='/home/tobi/cvcs/official/CVCS_project_DeepFake/xception_detection/weights/midway_best_non_cropped.pkl')
	main()