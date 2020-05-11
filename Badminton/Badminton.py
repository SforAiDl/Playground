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
import cv2

class Detector:
	def __init__(self, config_folder = "Badminton/config",config_path='Badminton/config/yolov3.cfg', weights_path='Badminton/config/yolov3.weights', class_path='Badminton/config/coco.names',img_size=416,conf_thres=0.8,nms_thres=0.4):
		self.config_path = config_path
		self.weights_path = weights_path
		self.class_path = class_path
		self.img_size = img_size
		self.conf_thres = conf_thres
		self.nms_thres = nms_thres
		self.config_folder = config_folder

	def detect_image(self,model,img,PIL_image_flag = True):

		if PIL_image_flag == False:
			# You may need to convert the color.
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = Image.fromarray(img)
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

	def detect_players_image(self,img_path=None,img=None,display_detection = True):

		# Load model and weights
		isFile = os.path.isfile(self.weights_path)
		if isFile == False:
			os.chdir(self.config_folder)
			print("Downloading the weights")
			try:
				os.system("./download_weights.sh")
			except:
				raise Exception("Not able to download the weights")
			os.chdir("../../")
		model = Darknet(self.config_path, img_size=self.img_size)
		model.load_weights(self.weights_path)
		model.cuda()
		model.eval()

		classes = self.load_classes(self.class_path)
		Tensor = torch.cuda.FloatTensor

		if img_path is None and img is None:
			print("Error!!! Enter either img_path or img")
			return [-1]
		elif img_path is not None and img is not None:
			print("Error!!! Enter only one out of img_path and img")
			return [-1]
		elif img_path is not None and img is None:
			print("Loading from image path")
			# load image and get detections
			self.img_path = img_path
			# img_path = "images/bad.jpg"
			prev_time = time.time()
			img = Image.open(self.img_path)
			detections = self.detect_image(model,img)
			inference_time = datetime.timedelta(seconds=time.time() - prev_time)
			# print ('Inference Time: %s' % (inference_time))
			# print(img.size)
			img = np.array(img)

			if display_detection == True:
				# Get bounding-box colors
				cmap = plt.get_cmap('tab20b')
				colors = [cmap(i) for i in np.linspace(0, 1, 20)]
				plt.figure()
				fig, ax = plt.subplots(1, figsize=(12,9))
				ax.imshow(img)

			pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
			pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
			unpad_h = self.img_size - pad_y
			unpad_w = self.img_size - pad_x

			flag = 0

			object_names = ['person']
			coordinate=[]
			if detections is not None and len(detections) < 4:
				# print("The objects detected are: ")
				unique_labels = detections[:, -1].cpu().unique()
				n_cls_preds = len(unique_labels)
				# browse detections and draw bounding boxes
				for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
					if classes[int(cls_pred)] in object_names:
						box_h = ((y2 - y1) / unpad_h) * img.shape[0]
						box_w = ((x2 - x1) / unpad_w) * img.shape[1]
						y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
						x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
						# print("\n##########################################################\n")
						# print("The box co-ordinates of " + str(classes[int(cls_pred)]) + " is :")
						# print("Top_left_x = " + str(x1.cpu().numpy()))
						# print("Top_left_y = " + str(y1.cpu().numpy()))
						# print("Height = " + str(box_h.cpu().numpy()))
						# print("Width = " + str(box_w.cpu().numpy()))
						coordinate.append(x1.cpu().numpy())
						coordinate.append(y1.cpu().numpy())
						coordinate.append(box_w.cpu().numpy())
						coordinate.append(box_h.cpu().numpy())

						flag = 1

						if display_detection == True:
							bbox_colors = random.sample(colors, n_cls_preds)
							color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
							bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
							ax.add_patch(bbox)
							plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
									bbox={'color': color, 'pad': 0})
			else:
				print("No objects of the desired type are detected!!\n")

			if flag == 0:
				print("None")
						
			print("\n##########################################################\n")

			# save image
			# plt.savefig(img_path.replace(".jpeg", "-det.jpeg"), bbox_inches='tight', pad_inches=0.0)
			if display_detection == True:
				plt.axis('off')
				plt.show()
			return coordinate

		elif img_path is None and img is not None:
			print("Loading from image")
			# Load model and weights
			isFile = os.path.isfile(self.weights_path)
			if isFile == False:
				os.chdir(self.config_folder)
				print("Downloading the weights")
				try:
					os.system("./download_weights.sh")
				except:
					raise Exception("Not able to download the weights")
				os.chdir("../../")
			model = Darknet(self.config_path, img_size=self.img_size)
			model.load_weights(self.weights_path)
			model.cuda()
			model.eval()

			classes = self.load_classes(self.class_path)
			Tensor = torch.cuda.FloatTensor

			# load image and get detections
			prev_time = time.time()
			detections = self.detect_image(model=model, img=img, PIL_image_flag=False)
			inference_time = datetime.timedelta(seconds=time.time() - prev_time)
			print ('Inference Time: %s' % (inference_time))
			# print(img.size)
			img = np.array(img)

			if display_detection == True:
				# Get bounding-box colors
				cmap = plt.get_cmap('tab20b')
				colors = [cmap(i) for i in np.linspace(0, 1, 20)]
				plt.figure()
				fig, ax = plt.subplots(1, figsize=(12,9))
				ax.imshow(img)

			pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
			pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
			unpad_h = self.img_size - pad_y
			unpad_w = self.img_size - pad_x

			flag = 0

			object_names = ['person']
			coordinate=[]

			if detections is not None and len(detections) < 4:
				print("The objects detected are: ")
				unique_labels = detections[:, -1].cpu().unique()
				n_cls_preds = len(unique_labels)
	
				# browse detections and draw bounding boxes
				for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
					if classes[int(cls_pred)] in object_names:
						box_h = ((y2 - y1) / unpad_h) * img.shape[0]
						box_w = ((x2 - x1) / unpad_w) * img.shape[1]
						y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
						x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
						# print("\n##########################################################\n")
						# print("The box co-ordinates of " + str(classes[int(cls_pred)]) + " is :")
						# print("Top_left_x = " + str(x1.cpu().numpy()))
						# print("Top_left_y = " + str(y1.cpu().numpy()))
						# print("Height = " + str(box_h.cpu().numpy()))
						# print("Width = " + str(box_w.cpu().numpy()))
						coordinate.append(x1.cpu().numpy())
						coordinate.append(y1.cpu().numpy())
						coordinate.append(box_w.cpu().numpy())
						coordinate.append(box_h.cpu().numpy())

						flag = 1
						
						if display_detection == True:
							bbox_colors = random.sample(colors, n_cls_preds)
							color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
							bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
							ax.add_patch(bbox)
							plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
									bbox={'color': color, 'pad': 0})
			else:
				print("No objects of the desired type are detected!!\n")

			if flag == 0:
				print("None")
						
			print("\n##########################################################\n")

			# save image
			# if save_detection == True:
			# 	plt.savefig(img_path.replace(".jpeg", "-det.jpeg"), bbox_inches='tight', pad_inches=0.0)

			if display_detection == True:
				plt.axis('off')
				plt.show()
			return coordinate


	def detect_players_video(self, video_path):
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)

		while(1):
			#reading the video frame by frame
			ret,frame = cap.read()
			if ret:
				(h, w) = frame.shape[:2]
				all_coordinates = self.detect_players_image(img=frame,display_detection = False)
				centerbottom = get_center_bottom(all_coordinates)
				for x in range(int(len(all_coordinates)/4)):
					start = 4*x
					end = start+4
					x1, y1, box_w, box_h = all_coordinates[start:end]
					cv2.rectangle(frame,(x1,y1),(x1+box_w,y1+box_h),(0,255,0),2)

				cv2.putText(frame, 'FPS: ' + str(fps),
										(int(w*0.004),int(h*0.04)),
										cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0), 3)
				cv2.imshow("Final output", frame)

			k = cv2.waitKey(1)
			if k == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
