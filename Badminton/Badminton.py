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
	def __init__(self, config_folder = "Badminton/config",config_path='Badminton/config/yolov3.cfg', weights_path='Badminton/config/yolov3.weights', class_path='Badminton/config/coco.names',img_size=416,conf_thres=0.8,nms_thres=0.4,tiny=False):
		self.config_path = config_path
		self.weights_path = weights_path
		self.class_path = class_path
		self.img_size = img_size
		self.conf_thres = conf_thres
		self.nms_thres = nms_thres
		self.config_folder = config_folder
		self.tiny = tiny

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
		# Tensor = torch.cuda.FloatTensor

		if torch.cuda.is_available():
			Tensor = torch.cuda.FloatTensor
			model.cuda()
		else:
			Tensor = torch.FloatTensor
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

	def detect_players_image(self,img_src,display_detection = True,save_detection=False,ret_img=False):

		if self.tiny == True:
			self.weights_path = 'Badminton/config/yolov3-tiny.weights'
			self.config_path = 'Badminton/config/yolov3-tiny.cfg'
		# Load model and weights
		isFile = os.path.isfile(self.weights_path)
		if isFile == False:
			os.chdir(self.config_folder)
			print("Downloading the weights")
			try:
				if self.tiny == False:
					os.system("bash download_weights.sh")
				else:
					os.system("bash download_tiny_weights.sh")
			except:
				raise Exception("Not able to download the weights")
			os.chdir("../../")
		model = Darknet(self.config_path, img_size=self.img_size)
		model.load_weights(self.weights_path)
		if torch.cuda.is_available():
			model.cuda()
			Tensor = torch.cuda.FloatTensor
		else:
			Tensor = torch.FloatTensor
		model.eval()

		classes = self.load_classes(self.class_path)
		# Tensor = torch.cuda.FloatTensor
		self.img_src = img_src
		prev_time = time.time()

		prev_time = time.time()

		if type(img_src) == str : #if input is image path
			img = Image.open(self.img_src)
		elif type(img_src) == np.ndarray : #if input is image array
			img = Image.fromarray(self.img_src)

		detections = self.detect_image(model,img)
		inference_time = datetime.timedelta(seconds=time.time() - prev_time)
		img = np.array(img)
		out_img = img.copy()

		#if display_detection == True:
			# Get bounding-box colors
		cmap = plt.get_cmap('tab20b')
		colors = [cmap(i) for i in np.linspace(0, 1, 20)]
			#plt.figure()
			#fig, ax = plt.subplots(1, figsize=(12,9))
			#ax.imshow(img)

		pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
		pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
		unpad_h = self.img_size - pad_y
		unpad_w = self.img_size - pad_x

		flag = 0

		object_names = ['person']
		coordinate=[]
		if detections is not None and len(detections) < 4:
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

					# if display_detection == True:
					# 	bbox_colors = random.sample(colors, n_cls_preds)
					# 	color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
					# 	bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
					# 	ax.add_patch(bbox)
					# 	plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',bbox={'color': color, 'pad': 0})
					# else:	
					label = classes[int(cls_pred)]
					bbox_colors = random.sample(colors, n_cls_preds)
					#color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
					color = tuple([255*x for x in bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]])
					cv2.putText(img=out_img, text=label, org=(x1, y1 - 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
					cv2.rectangle(out_img, (x1, y1), (x1 + box_w, y1 + box_h),(128,0,128), 2) #purple bbox 
						
		else:
			print("No objects of the desired type are detected!!\n")

		if flag == 0:
			print("None")
						
		# save image
		# plt.savefig(img_path.replace(".jpeg", "-det.jpeg"), bbox_inches='tight', pad_inches=0.0)
		if display_detection == True:

			print("\n##########################################################\n")
			cv2.imshow("Final output", out_img)
			#plt.axis('off')
			#plt.show()
		if save_detection == True:
			if type(img_src) == str:
				print("Output image can be found here: " + img_src.replace(".jpg", "-out.jpg"))
				cv2.imwrite(img_src.replace(".jpg", "-out.jpg"),cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
			else:
				print("Output image can be found here: " + os.getcwd()+"/output.jpg")
				cv2.imwrite(os.getcwd()+"/output.jpg",cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
			
		
		if not ret_img :
			#cv2.imshow("Final output", out_img)
			return None,None
		else :
			return out_img,coordinate


	def detect_players_video(self, video_path):
	
		out_video = []
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		prev_time2 = time.time()
		while(1):
			#reading the video frame by frame
			ret,frame = cap.read()
			if not ret:
				break

			(h, w) = frame.shape[:2]
			out_frame,all_coordinates = self.detect_players_image(frame,ret_img=1,display_detection=False)
			centerbottom = get_center_bottom(all_coordinates)
			out_video.append(out_frame)
			# k = cv2.waitKey(1)
			# if k == ord('q'):
			# 	break

		cap.release()
		print("Time taken is:" + str(time.time() - prev_time2))
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		out = cv2.VideoWriter(video_path.replace(".mp4", "-out.mp4"),fourcc,fps,(w,h))
		for i in range(len(out_video)):
			out.write(out_video	[i])
		print("Output Video can be found here: " + video_path.replace(".mp4", "-out.mp4"))
		out.release()
		cv2.destroyAllWindows()

