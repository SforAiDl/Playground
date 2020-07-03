import datetime
import os
import random
import sys
import subprocess
import time

import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
import numpy as np

from Football.utils import *
from Badminton.utilities.datasets import *
from Badminton.utilities.parse_config import *
from Badminton.utilities.models import *


class Detector:
    def __init__(self,
                 config_folder="config",
                 config_path='config/yolov3.cfg',
                 weights_path='config/yolov3.weights',
                 class_path='config/coco.names',
                 img_size=416,
                 conf_thres=0.6,
                 nms_thres=0.4,
                 tiny=False,
                 Windows=False):

        self.config_path = config_path
        self.weights_path = weights_path
        self.class_path = class_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.config_folder = config_folder
        self.tiny = tiny
        self.Windows = Windows

    def detect_image(self, model, img):

        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.img = img

        # scale and pad image
        ratio = min(self.img_size/self.img.size[0], self.img_size/self.img.size[1])
        imw = round(self.img.size[0] * ratio)
        imh = round(self.img.size[1] * ratio)
        img_transforms = transforms.Compose([
            transforms.Resize((imh, imw)),
            transforms.Pad((max(int((imh-imw)/2), 0),
                            max(int((imw-imh)/2), 0),
                            max(int((imh-imw)/2), 0),
                            max(int((imw-imh)/2), 0)),
                            (128, 128, 128)),
                            transforms.ToTensor()])
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

        with torch.no_grad():
            detections = model(input_img)
            detections = non_max_suppression(
                detections, 80, self.conf_thres, self.nms_thres)
        return detections[0]

    def load_classes(self, path):
        """
        Loads class labels at 'path'
        """
        fp = open(path, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def detect_players_image(self,
                             img_src,
                             display_detection=True,
                             save_detection=False,
                             ret_img=False):

        if self.tiny:
            self.weights_path = 'config/yolov3-tiny.weights'
            self.config_path = 'config/yolov3-tiny.cfg'

        isFile = os.path.isfile(self.weights_path)
        if not isFile:
            os.chdir(self.config_folder)
            print("Downloading the weights")
            try:
                if not self.Windows:
                    if not self.tiny:
                        os.system("bash download_weights.sh")
                    else:
                        os.system("bash download_tiny_weights.sh")
                else:
                    if not self.tiny:
                        os.system("powershell.exe download_weights.ps1")
                    else:
                        os.system("powershell.exe download_tiny_weights.ps1")

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

        self.img_src = img_src
        prev_time = time.time()

        if isinstance(img_src, str):
            # if input is image path
            img = cv2.imread(self.img_src)
        elif isinstance(img_src, np.ndarray):
            # if input is image array
            img = Image.fromarray(self.img_src)

        detections = self.detect_image(model, img)
        inference_time = datetime.timedelta(seconds=time.time() - prev_time)
        img = np.array(img)
        out_img = img.copy()

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        pad_x = max(img.shape[0] - img.shape[1], 0) * \
            (self.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * \
            (self.img_size / max(img.shape))
        unpad_h = self.img_size - pad_y
        unpad_w = self.img_size - pad_x

        flag = 0

        object_names = ['person','sports ball']
        coordinate = []
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if classes[int(cls_pred)] in object_names:
                    box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                    box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                    y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                    x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                    coordinate.append(x1.cpu().numpy())
                    coordinate.append(y1.cpu().numpy())
                    coordinate.append(box_w.cpu().numpy())
                    coordinate.append(box_h.cpu().numpy())

                    flag = 1

                    label = classes[int(cls_pred)]
                    bbox_colors = random.sample(colors, n_cls_preds)
                    color = tuple([255*x for x in bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]])
                    cv2.putText(img=out_img,
                                text=label,
                                org=(x1, y1 - 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(255,255,255),
                                thickness=2)
                    cv2.rectangle(out_img,
                                 (x1, y1),
                                 (x1 + box_w, y1 + box_h),
                                 (128,0,128),
                                 2)  # purple bbox

        else:
            pass



        if display_detection:
        	cv2.imshow('Final ouput', cv2.resize(out_img,(960, 540)))
        	cv2.waitKey(0)
        	cv2.destroyAllWindows()

        if save_detection:
            if isinstance(img_src, str):
                print("Output image can be found here: " +
                      img_src.replace(".jpg", "-out.jpg"))
                cv2.imwrite(img_src.replace(".jpg", "-out.jpg"),cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            else:
                print("Output image can be found here: " + os.getcwd()+"/output.jpg")
                cv2.imwrite(os.getcwd()+"/output.jpg",cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))


        if not ret_img:
            return None,None
        else:
            return out_img, coordinate

    def detect_players_video(self, video_path, optimization=False,
                                frames_skipped_input=1, heatmap=False, output_path=None):

            out_video = []
            cap = cv2.VideoCapture(video_path)
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fps = cap.get(cv2.CAP_PROP_FPS)
            start_time = time.time()

            print("Optimization set to %s" %str(optimization))
            frames_skipped = frames_skipped_input

            # initialize a dictionary that maps strings to their corresponding
            # OpenCV object tracker implementations
            OPENCV_OBJECT_TRACKERS = {
                "csrt": cv2.TrackerCSRT_create,
                "kcf": cv2.TrackerKCF_create,
                "boosting": cv2.TrackerBoosting_create,
                "mil": cv2.TrackerMIL_create,
                "tld": cv2.TrackerTLD_create,
                "medianflow": cv2.TrackerMedianFlow_create,
                "mosse": cv2.TrackerMOSSE_create
            }
            # initialize OpenCV's special multi-object tracker
            trackers = cv2.MultiTracker_create()
            tracker = OPENCV_OBJECT_TRACKERS['csrt']()
            print("Using Tracker:", tracker.__class__.__name__)
            pbar = tqdm(total=total_frame)
            list_of_all_coordinates = []
            i = 0
            frame_count = 0
            while(1):
                 # Read video frame
                 ret, frame = cap.read()
                 frame_count += 1
                 if (frame_count % 20 == 0):
                    i = 0
                 if not ret:
                    break
                 else:
                    (h, w) = frame.shape[:2]
                    if i == 0:
                        print("Running Detector")
                        frame, all_coordinates = self.detect_players_image(frame,
                                                                        ret_img=1,
                                                                        display_detection=False)
                        while i < len(all_coordinates):
                            # Every object detected has 4 box properties: x, y, w, h
                            trackers.add(tracker, frame, tuple(all_coordinates[i:i+4]))
                            i += 4
                        print(len(all_coordinates) // 4, "objects added to tracker.")
                    else:
                        ok, bbox = trackers.update(frame)
                        if not ok:
                            print("Tracking Failed")
                            i = 0
                        else:
                            for newbox in bbox:
                                p1 = (int(newbox[0]), int(newbox[1]))
                                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                                cv2.rectangle(frame, p1, p2, (200,0,0))

                    if heatmap:
                        list_of_all_coordinates.append(all_coordinates)
                    else:
                        out_video.append(frame)
#                        cv2.imshow("Frame", frame)

                    pbar.update(1)
                    keyPress = cv2.waitKey(1)
                    if keyPress != -1:
                        print("Key Pressed:", keyPress)
                    if keyPress == ord('q'):
                        break
                    if keyPress == ord('r'):
                        print("r pressed, resetting detector")
                        trackers = cv2.MultiTracker_create()
                        i = 0

            cap.release()
            pbar.close()
            print("Video Analysed in :" + str(time.time() - start_time) + 's')

            if heatmap:
                return list_of_all_coordinates
            else:
                # ------------------------------
                # -----------  TODO  -----------
                # ------------------------------
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                if output_path is None:
                    output_path = video_path.replace(".mp4", "-out.avi")
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                for i in range(len(out_video)):
                    out.write(out_video[i])
                print("Output Video saved here: " + output_path)
                out.release()
                cv2.destroyAllWindows()



