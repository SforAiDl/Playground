# Docs for Badminton Module

## Documentation for Class Badminton.Badminton.Detector
### Badminton.Badminton.Detector.detect_players_image(self,img_src,display_detection = True,ret_img=False)
   Detects players in the Image.  
   Takes as arguments img_src,display_detection, ret_img and by default has no return value and displays the image with players detected(if any) by bounding boxes. If ret_img is set to True, an output image with bounding boxes is returned instead of displaying it.  
   If no objects of the desired type are detected a corresponding message is displayed. 

#### Parameters
* **img_src**: str or array - 
   Image source either as an image path or image array
* **display_detection**: bool, optional, default:True - 
   Flag which tells whether the detection with bounding boxes is to be displayed.
* **ret_img**: bool, optional, default:False - 
   Flag which tells whether an image is to be returned. If set to True, method returns Output image and coordinates of detected object.

#### Returns
   default: None, None

#### Usage
```python
from Badminton.Badminton import Detector
detector_object = Detector()
out_frame,all_coordinates = detector_object.detect_players_image(img,ret_img=1,display_detection=False)
```


### Badminton.Badminton.Detector.detect_players_video(self, video_path, optimization=False, frames_skipped_input=1)
   Detects players in video. Replaces the video file with a video with bounding boxes over detected locations of players over the original video. If optimization is True, an optimized algorithm is used to detect the players in the video and the time required is reduced by a factor of frames_skipped_input

#### Parameters
* **video_path**: str - 
   Path of the video file as a string.  
* **optimization**: bool, optional, default:False - 
   Determines whether optimized version has to be used or not
* **frames_skipped**: int, optional, default: 1 - 
   Number of frames skipped. Default value is 1 if optimization is False indicating that no frames will be skipped. Default value is 3 if tiny is True for the Detector class and 5 if tiny is False.

#### Returns
   default: None, None

#### Usage
```python
from Badminton.Badminton import Detector
obj = Detector(tiny=False)
obj.detect_players_video("Badminton/images/video2.mp4",optimization=True,frames_skipped_input=5)
```


### Badminton.Badminton.Detector.get_heatmap(self, video_path)
Detects players and tracks their respective movements in the video; accordingly generates a picture plotting the heatmap of the players.

#### Parameters
* **video_path**: str - 
   Path of the video file as a string.

#### Returns
   default: None, None

#### Usage
```python
from Badminton.Badminton import Detector
obj = Detector(tiny=True)
obj.get_heatmap("Badminton/images/video2.mp4")
```