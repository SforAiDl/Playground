# Docs for Badminton Module

## Documentation for Class Badminton.Badminton.Detector
### Badminton.Badminton.Detector.detect_players_image(self,img_src,display_detection = True,ret_img=False)
   Detects players in the Image.  
   Takes as arguments img_src,display_detection, ret_img and by default has no return value and displays the image with players detected(if any) by bounding boxes. If ret_img is set to True, an output image with bounding boxes is returned instead of displaying it.  
   If no objects of the desired type are detected a coressponding message is displayed. 

#### Parameters
* **img_src**: str or array
   image source either as an image path or image array
* **display_detection**: bool, optional, default:True
   Flag which tells whether the detection with bounding boxes is to be displayed.
* **ret_img**: bool, optional, default:False
   Flag which tells whether an image is to be returned. If set to True, method returns Output image and coordinates of detected object.

#### Returns
   default: None, None

#### Usage
```python
from Badminton.Badminton import Detector
detector_object = Detector()
out_frame,all_coordinates = detector_object.detect_players_image(img,ret_img=1,display_detection=False)
```

### Badminton.Badminton.Detector.detect_players_video(self, video_path)
   Detects players in video. Replaces the video file with a video with bounding boxes over detected locations of players over the original video.  

#### Parameters
* **video_path**: str
   Path of the video file as a string.  

#### Usage
```python
from Badminton.Badminton import Detector
obj = Detector(tiny=True)
obj.detect_players_video("Badminton/images/video.mp4")
```
