from Badminton.utilities.utils import *
from Badminton.utilities.datasets import *
from Badminton.utilities.parse_config import *
from Badminton.utilities.models import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class Detector:
	def __init__(self, config_path='Badminton/config/yolov3.cfg', weights_path='Badminton/config/yolov3.weights', class_path='Badminton/config/coco.names',img_size=416,conf_thres=0.8,nms_thres=0.4):
		self.config_path = config_path
		self.weights_path = weights_path
		self.class_path = class_path
		self.img_size = img_size
		self.conf_thres = conf_thres
		self.nms_thres = nms_thres

	def detect_image(self,model,img):
		self.img = img

		# scale and pad image
		ratio = min(self.img_size/self.img.size[0], self.img_size/self.img.size[1])
		imw = round(self.img.size[0] * ratio)
		imh = round(self.img.size[1] * ratio)
		img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
			 transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
							(128,128,128)),
			 transforms.ToTensor(),
			 ])
		# convert image to Tensor
		image_tensor = img_transforms(self.img).float()
		image_tensor = image_tensor.unsqueeze_(0)
		Tensor = torch.cuda.FloatTensor
		input_img = Variable(image_tensor.type(Tensor))
		# run inference on the model and get detections
		with torch.no_grad():
			detections = model(input_img)
			detections = non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
		return detections[0]

	def load_classes(self,path):
		"""
		Loads class labels at 'path'
		"""
		fp = open(path, "r")
		names = fp.read().split("\n")[:-1]
		return names

	def detect_players(self,img_path):
		# Load model and weights
		model = Darknet(self.config_path, img_size=self.img_size)
		model.load_weights(self.weights_path)
		model.cuda()
		model.eval()

		classes = self.load_classes(self.class_path)
		Tensor = torch.cuda.FloatTensor

		# load image and get detections
		self.img_path = img_path
		# img_path = "images/bad.jpg"
		prev_time = time.time()
		img = Image.open(self.img_path)
		detections = self.detect_image(model,img)
		inference_time = datetime.timedelta(seconds=time.time() - prev_time)
		print ('Inference Time: %s' % (inference_time))

		# Get bounding-box colors
		cmap = plt.get_cmap('tab20b')
		colors = [cmap(i) for i in np.linspace(0, 1, 20)]

		img = np.array(img)
		plt.figure()
		fig, ax = plt.subplots(1, figsize=(12,9))
		ax.imshow(img)

		pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
		pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
		unpad_h = self.img_size - pad_y
		unpad_w = self.img_size - pad_x

		flag = 0

		object_names = ['person']

		if detections is not None:
			print("The objects detected are: ")
			unique_labels = detections[:, -1].cpu().unique()
			n_cls_preds = len(unique_labels)
			bbox_colors = random.sample(colors, n_cls_preds)
			# browse detections and draw bounding boxes
			for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
				if classes[int(cls_pred)] in object_names:
					box_h = ((y2 - y1) / unpad_h) * img.shape[0]
					box_w = ((x2 - x1) / unpad_w) * img.shape[1]
					y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
					x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
					print("\n##########################################################\n")
					print("The box co-ordinates of " + str(classes[int(cls_pred)]) + " is :")
					print("Centre_x = " + str(x1.cpu().numpy()))
					print("Centre_y = " + str(y1.cpu().numpy()))
					print("Height = " + str(box_h.cpu().numpy()))
					print("Width = " + str(box_w.cpu().numpy()))
					flag = 1
					color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
					bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
					ax.add_patch(bbox)
					plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
							bbox={'color': color, 'pad': 0})
		else:
			print("No objects of the desired type are detected!!\n")

		if flag == 0:
			print("None")
					
		plt.axis('off')
		print("\n##########################################################\n")

		# save image
		# plt.savefig(img_path.replace(".jpeg", "-det.jpeg"), bbox_inches='tight', pad_inches=0.0)
		plt.show()