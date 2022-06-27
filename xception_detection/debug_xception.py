import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset	
def main():
	args = parse.parse_args()
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	#model.load_state_dict(torch.load(model_path, map_location='cpu'))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model.eval()
	with torch.no_grad():
		image = torch.randn((32, 3, 299, 299), dtype=torch.float)
		print(image.size())
		outputs = model(image)
if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--batch_size', '-bz', type=int, default=32)
	parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Deepfakes_c0_test.txt')
	main()
	print('Hello world!!!')