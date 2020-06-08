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

from Badminton.utilities.utils import *
from Badminton.utilities.datasets import *
from Badminton.utilities.parse_config import *
from Badminton.utilities.models import *


class Detector:
    def __init__(self,
                 config_folder="Badminton/config",
                 config_path='Badminton/config/yolov3.cfg',
                 weights_path='Badminton/config/yolov3.weights',
                 class_path='Badminton/config/coco.names',
                 img_size=416,
                 conf_thres=0.8,
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

        Tensor = if_cuda_is_available(model)
        
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
            self.weights_path = 'Badminton/config/yolov3-tiny.weights'
            self.config_path = 'Badminton/config/yolov3-tiny.cfg'

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
        Tensor = if_cuda_is_available(model)
        model.eval()
       

        classes = self.load_classes(self.class_path)

        self.img_src = img_src
        prev_time = time.time()

        if isinstance(img_src, str):
            check_file_exists(img_src)
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

        object_names = ['person']
        coordinate = []
        if detections is not None and len(detections) < 4:
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
        	cv2.imshow('Final ouput', out_img)
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



    def detect_players_video(self,
                             video_path,
                             optimization=False,
                             frames_skipped_input=1, heatmap=False):
        check_file_exists(video_path)

        out_video = []
        cap = cv2.VideoCapture(video_path)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_time2 = time.time()

        if heatmap:
            list_of_all_coordinates=[]
            
        if optimization:
            count_of_frames=0
            frames_skipped=get_frames_skipped(self.tiny, frames_skipped_input)
            print("\nOptimization is True")
            no_frame_read = 1
            while(1):
                # reading the video frame by frame
                ret,frame = cap.read()
                prev_time1 = time.time()

                if not ret:
                    break
                
                (h, w) = frame.shape[:2]

                
                #   The idea is that we will only detect_players_image after every 5 or 'frames_skipped' frames and
                #   for the other frames we will calculate the weighted average of the previous location and next location to determine the position of the players

                if count_of_frames % frames_skipped == 0:
                    out_frame, all_coordinates = self.detect_players_image(frame,
                                                                           ret_img=1,
                                                                           display_detection=False)
                    #   out_frame conatins the new frame with the players detected and all_cooridnates contains the coordinates of the box(es)

                    if(no_frame_read==1):   #   This snippet is for the progress bar
                        pbar=tqdm(total=total_frame)
                    
                    if count_of_frames==0: #for the first frame
                        frame_list=[] #initialize a frame list (size will be frames_skipped)
                        for f in range(frames_skipped):
                            frame_list.append(frame)
                        #ensure first frame detects 2 players-->
                        if len(all_coordinates)!= 8:
                            all_coordinates.append(10)
                            all_coordinates.append(10)
                            all_coordinates.append(10)
                            all_coordinates.append(10)
                        previous_frame_coordiantes = all_coordinates
                        if heatmap:
                            current_coords_list=all_coordinates
                    else:
                        #for every frame read thereafter
                        current_coords_list = check_if_two_players_detected(previous_frame_coordiantes,
                                                                                 all_coordinates)
                        step_list = calculate_step(previous_frame_coordiantes,
                                                        current_coords_list,
                                                        frames_skipped)
                        for frame_no in range(1, frames_skipped):
                            frame_coords = get_frame_coords(previous_frame_coordiantes,
                                                                 step_list,
                                                                 frame_no)
                            frame_list[frame_no] = draw_boxes(frame_list[frame_no],
                                                                   frame_coords)
                            out_video.append(frame_list[frame_no])
                            if heatmap:
                                list_of_all_coordinates.append(current_coords_list)
                        previous_frame_coordiantes = current_coords_list
                    out_video.append(out_frame)
                    if heatmap:
                        list_of_all_coordinates.append(current_coords_list)
                    frame_list[0] = out_frame

                    current_time = time.time()
                    eta = time_elapsed_frame_left(current_time,prev_time1,total_frame,no_frame_read)
                    no_frame_read += frames_skipped
                    pbar.update(frames_skipped)
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        break


                else:   #  For all the other frames that we have skipped
                    frame_list[count_of_frames % frames_skipped] = frame    #   Add the skipped frames to the frame_list so that they can be modified with the boxes 
                
                count_of_frames+=1
        
        else:
           #   IF NO OPTIMIZATION
            if heatmap:
                list_of_all_coordinates=[]
            print("\nOptimization is False")
            frames_skipped=1
            print("detect_players_image is run in the video every",frames_skipped, "frames\n")

            no_frame_read = 1
            while(1):
                #reading the video frame by frame
                ret,frame = cap.read()
                prev_time1 = time.time()

                if not ret:
                    break

                (h, w) = frame.shape[:2]
                out_frame,all_coordinates = self.detect_players_image(frame,
                                                                      ret_img=1,
                                                                      display_detection=False)
                # centerbottom = get_center_bottom(all_coordinates)
                out_video.append(out_frame)
                if heatmap:
                    list_of_all_coordinates.append(all_coordinates)
                current_time = time.time()
                eta = time_elapsed_frame_left(current_time,prev_time1,total_frame,no_frame_read)
                if(no_frame_read == 1):
                    pbar=tqdm(total = total_frame)
                no_frame_read += 1
                pbar.update(1)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    break

        cap.release()
        pbar.close()
        print("Time taken for reading video is:" + str(time.time() - prev_time2))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        
        if not heatmap:
            out = cv2.VideoWriter(video_path.replace(".mp4", "-out.mp4"), fourcc, fps, (w, h))
            for i in range(len(out_video)):
                out.write(out_video[i])
            print("Output Video can be found here: " + video_path.replace(".mp4", "-out.mp4"))
        else:
            return list_of_all_coordinates
        out.release()
        cv2.destroyAllWindows()

    def get_heatmap(self,
                    video_path,
                    optimization=False,
                    frames_skipped_input=1):
        check_file_exists(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0

        #read first frame for image
        ret, frame= cap.read()

        matrix,template = initialize_court(frame)
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(template)
        print(frame_count)

        #initialize cap again so that the first frame is not skipped in detect_players_video
        cap = cv2.VideoCapture(video_path)

        coords=self.detect_players_video(video_path=video_path, optimization=optimization, frames_skipped_input=frames_skipped_input,heatmap=True)

        for coordinate_n in coords:
            centerbottom=get_center_bottom(coordinate_n)
            #need to insert code for tqdm for heatmap part
            #video tqdm is already being shown when we call detect_players_video method
            ax=get_transformed_bbox(centerbottom, matrix, ax)        
        cv2.destroyAllWindows()
        plt.savefig('./Badminton/images/heatmap.png', bbox_inches='tight')
        plt.show()
