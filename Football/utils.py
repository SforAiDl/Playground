from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

#   This method draws boxes on the frame 'frame_to_draw' according to the coordinated in 'coord_list_inp'
def draw_boxes(frame_to_draw, coord_list_inp):
    cv2.putText(img=frame_to_draw, text="person", org=(coord_list_inp[0], coord_list_inp[1] - 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
    cv2.rectangle(frame_to_draw, (coord_list_inp[0], coord_list_inp[1]), (coord_list_inp[0] + coord_list_inp[2], coord_list_inp[1] + coord_list_inp[3]),(128,0,128), 2) #purple bbox
    cv2.putText(img=frame_to_draw, text="person", org=(coord_list_inp[4], coord_list_inp[5] - 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
    cv2.rectangle(frame_to_draw, (coord_list_inp[4], coord_list_inp[5]), (coord_list_inp[4] + coord_list_inp[6], coord_list_inp[5] + coord_list_inp[7]),(128,0,128), 2) #purple bbox
    return frame_to_draw

#   This method will calculate the single step by which the coordinates of the skipped frames should be updated (the update value)
def calculate_step(prev_coords, current_coords, num_frames_skipped):
    step = [0]*8
    for i in range(len(prev_coords)):
        step[i]=(current_coords[i]- prev_coords[i])/num_frames_skipped
    return step

#   This method will return the coordinates of the players based on the previous coordinates by adding the step n number of times (n being the no of frames after the frame of the 'coords')
def get_frame_coords(coords, step, no_of_steps):
    new_coords = [0]*8
    for i in range(len(coords)):
        new_coords[i]=int(coords[i]+(step[i]*no_of_steps))
    return new_coords

#   This method will verify if 2 players have been detected in 'current_coords'; if not, then it will find the player who hasn't been detected and uses its previous frame's coordinates 'prev_coords' and assumes its position didn't change in the current frame
def check_if_two_players_detected(prev_coords, current_coords):
    if len(current_coords)!= 8:
        # find which player is not detected
        if len(current_coords)==4:
            diff_player1=abs(current_coords[0]-prev_coords[0])+abs(current_coords[1]-prev_coords[1])
            diff_player2=abs(current_coords[0]-prev_coords[4])+abs(current_coords[1]-prev_coords[5])
            if diff_player2 > diff_player1:
                new_current_coords=[current_coords[0], current_coords[1], current_coords[2], current_coords[3], prev_coords[4],prev_coords[5], prev_coords[6], prev_coords[7]]
            else:
                new_current_coords=[prev_coords[0], prev_coords[1], prev_coords[2], prev_coords[3], current_coords[0],current_coords[1], current_coords[2], current_coords[3]]
        else:
            new_current_coords=prev_coords
    else:
        new_current_coords=current_coords
    return new_current_coords

positions=[]
positions2=[]
count =0 

def heatmap_template(result):
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -5)
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    #Horizontal morphology
    cols = horizontal.shape[1]
    horizontal_size = cols // 3
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal=cv2.morphologyEx(horizontal,cv2.MORPH_OPEN,horizontalStructure,iterations=1)
    horizontal = cv2.dilate(horizontal,horizontalStructure,iterations=5)
    #Vertical morphology
    rows = vertical.shape[0]
    verticalsize = rows // 5
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical=cv2.morphologyEx(vertical,cv2.MORPH_OPEN,verticalStructure,iterations=1)
    vertical = cv2.dilate(vertical,verticalStructure,iterations=2)

    final = vertical+horizontal
    final[(final.shape[0]//2)-5:(final.shape[0]//2)+5,:]=255 #middle net line
    
    green = np.zeros((result.shape), np.uint8)
    green[:,:,:] = (np.mean(result[:,:,0]),np.mean(result[:,:,1]),np.mean(result[:,:,2]))
    final_inv = cv2.bitwise_not(final)
    template = cv2.bitwise_and(~green,~green,mask = final_inv)
    
    return ~template


def draw_circle(event, x, y, flags, image):

    global positions,count

    # If left button is clicked then store the position where pointer is
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(image, (x, y), 2, (0, 0, 255), 2)
        positions.append([x, y])
        if count != 3:
            positions2.append([x, y])
        elif count == 3:
            positions2.insert(2, [x, y])
        count += 1
    return positions


def get_court_coordinates(image):
    print("\n##########################################################\n")
    print("Select 4 corners in Top Left, Top Right, Bottom Left & Bottom Right in order and then press escape")
    print("\n##########################################################\n")
    cv2.namedWindow('Select 4 corners in Top Left, Top Right, Bottom Left & Bottom Right in order and then press escape',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select 4 corners in Top Left, Top Right, Bottom Left & Bottom Right in order and then press escape', draw_circle,image)
               
    while True:
        cv2.imshow('Select 4 corners in Top Left, Top Right, Bottom Left & Bottom Right in order and then press escape', image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or (len(positions) > 0 and len(positions) == 4):
            break

    cv2.destroyAllWindows()

def PIL_to_OpenCV(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR 
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)    
    return img

def get_center_bottom(coordinate):
    centerbottom = []
    if coordinate is not None:
        for x in range(int(len(coordinate)/4)):
            start = 4*x
            end = start+4
            x1, y1, box_w, box_h = coordinate[start:end]
            x_centre = x1 + (float(box_w/2))
            y_bottom = y1 + box_h
            centerbottom.append(x_centre)
            centerbottom.append(y_bottom)
        return centerbottom
    else:
        return None

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])

def draw_boxes(frame_to_draw, coord_list_inp):
    cv2.putText(img=frame_to_draw, text="person", org=(coord_list_inp[0], coord_list_inp[1] - 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
    cv2.rectangle(frame_to_draw, (coord_list_inp[0], coord_list_inp[1]), (coord_list_inp[0] + coord_list_inp[2], coord_list_inp[1] + coord_list_inp[3]),(128,0,128), 2) #purple bbox
    cv2.putText(img=frame_to_draw, text="person", org=(coord_list_inp[4], coord_list_inp[5] - 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
    cv2.rectangle(frame_to_draw, (coord_list_inp[4], coord_list_inp[5]), (coord_list_inp[4] + coord_list_inp[6], coord_list_inp[5] + coord_list_inp[7]),(128,0,128), 2) #purple bbox
    return frame_to_draw

def calculate_step(prev_coords, current_coords, num_frames_skipped):
    step=[0]*len(prev_coords)
    for i in range(len(prev_coords)):
        try:
            step[i]=(current_coords[i]- prev_coords[i])/num_frames_skipped
        except:
            continue
    return step

def get_frame_coords(coords, step, no_of_steps):
    new_coords=[0]*len(coords)
    for i in range(len(coords)):
        new_coords[i]=int(coords[i]+(step[i]*no_of_steps))
    return new_coords

def check_if_two_players_detected(prev_coords, current_coords):
    if len(current_coords)!= 8:
        # find which player  is not detected
        if len(current_coords)==4:
            diff_player1=abs(current_coords[0]-prev_coords[0])+abs(current_coords[1]-prev_coords[1])
            diff_player2=abs(current_coords[0]-prev_coords[4])+abs(current_coords[1]-prev_coords[5])
            if diff_player2 > diff_player1:
                new_current_coords=[current_coords[0], current_coords[1], current_coords[2], current_coords[3], prev_coords[4],prev_coords[5], prev_coords[6], prev_coords[7]]
            else:
                new_current_coords=[prev_coords[0], prev_coords[1], prev_coords[2], prev_coords[3], current_coords[0],current_coords[1], current_coords[2], current_coords[3]]
        else:
            new_current_coords=prev_coords
    else:
        new_current_coords=current_coords
    return new_current_coords

