import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from sklearn.utils import class_weight
import numpy as np
from models import model_selection
from mesonet import Meso4, MesoInception4
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
from matplotlib import pyplot as plt

def main():
	args = parse.parse_args()
	name = args.name
	continue_train = args.continue_train
	train_list = args.train_list
	val_list = args.val_list
	epoches = args.epoches
	batch_size = args.batch_size
	model_name = args.model_name
	model_path = args.model_path
	output_path = os.path.join('./output', name)
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	torch.backends.cudnn.benchmark=True
	train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])
	val_dataset = MyDataset(txt_path=val_list, transform=xception_default_data_transforms['val'])
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
	train_dataset_size = len(train_dataset)
	val_dataset_size = len(val_dataset)
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	if continue_train:
		model.load_state_dict(torch.load(model_path, map_location='cpu'))
	model = model.cuda()
	#realc, fakec = train_dataset.get_counts()
	#class_weights = [realc, fakec]
	labels = []
	for _, targets in train_loader:
		for target in targets:
			labels.append(int(target))
	classes = np.arange(2)
	class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=np.array(labels))
	class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
	criterion = nn.CrossEntropyLoss(class_weights)
	optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
	# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.05)
	model = nn.DataParallel(model)
	best_model_wts = model.state_dict()
	best_acc = 0.0
	iteration = 0
	epoch_losses = []
	epoch_accs = []
	for epoch in range(epoches):
		print('Epoch {}/{}'.format(epoch+1, epoches))
		print('-'*10)
		model.train()
		train_loss = 0.0
		train_corrects = 0.0
		val_loss = 0.0
		val_corrects = 0.0

		for (image, labels) in train_loader:
			iter_loss = 0.0
			iter_corrects = 0.0
			image = image.cuda()
			labels = labels.cuda()
			optimizer.zero_grad()
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			iter_loss = loss.data.item()
			train_loss += iter_loss
			iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
			train_corrects += iter_corrects
			iteration += 1
			if not (iteration % 100): #enter only if multiple of 20
				#the dataset is strongly unbalanced, so we need to check some others metrics like f1-score
				#this_iter_acc = iter_corrects / batch_size
				#this_iter_f1 = f1_loss(outputs, labels)
				print('iteration {} train loss: {:.4f} Acc: {:.4f} '.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))


		epoch_loss = train_loss / train_dataset_size
		epoch_acc = train_corrects / train_dataset_size
		print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
		true_one, true_zero, false_one, false_zero = 0, 0, 0, 0
		model.eval()
		with torch.no_grad():
			for (image, labels) in val_loader:
				image = image.cuda()
				labels = labels.cuda()
				outputs = model(image)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				val_loss += loss.data.item()
				val_corrects += torch.sum(preds == labels.data).to(torch.float32)
				true_zero += torch.sum((preds == labels.data) * (preds == 0)).to(torch.float32)
				false_zero += torch.sum((preds != labels.data) * (preds == 0)).to(torch.float32)
				true_one += torch.sum((preds == labels.data) * (preds == 1)).to(torch.float32)
				false_one += torch.sum((preds != labels.data) * (preds == 1)).to(torch.float32)
			epoch_loss = val_loss / val_dataset_size
			epoch_acc = val_corrects / val_dataset_size
			epoch_precision1 = true_one / (true_one+false_one)
			epoch_precision0 = true_zero / (true_zero+false_zero)
			epoch_recall1 = true_one / (true_one+false_zero)
			epoch_recall0 = true_zero / (true_zero+false_one)
			epoch_losses.append(epoch_loss)
			epoch_accs.append(epoch_acc.item())
			print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
			print('epoch precision: {:.4f} for class 1 {:.4f} for class 0'.format(epoch_precision1, epoch_precision0))
			print('epoch recall: {:.4f} for class 1 {:.4f} for class 0'.format(epoch_recall1, epoch_recall0))

			if epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()
		scheduler.step()

		#if not (epoch % 40):
		torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
	"""Loss representation:"""
	x = np.linspace(0, 10, epoches)
	plt.figure(0)
	plt.plot(x, epoch_losses, color='red')
	plt.title('{}_losses'.format(model_name))
	plt.show()
	plt.figure(1)
	plt.plot(x, epoch_accs, color='blue')
	plt.title('{}_accuracy'.format(model_name))
	plt.show()
	print('Best val Acc: {:.4f}'.format(best_acc))
	model.load_state_dict(best_model_wts)
	torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))


def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--name', '-n', type=str, default='fs_xception_c0_299')
	parse.add_argument('--train_list', '-tl' , type=str, default = './data_list/FaceSwap_c0_train.txt')
	parse.add_argument('--val_list', '-vl' , type=str, default = './data_list/FaceSwap_c0_val.txt')
	parse.add_argument('--batch_size', '-bz', type=int, default=64)
	parse.add_argument('--epoches', '-e', type=int, default='18')
	parse.add_argument('--model_name', '-mn', type=str, default='sgd_steplr0_005_balanced_data.pkl')
	parse.add_argument('--continue_train', type=bool, default=False)
	parse.add_argument('--model_path', '-mp', type=str, default='./output/df_xception_c0_299/1_df_c0_299.pkl')
	main()
