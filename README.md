# Playground
A python library consisting of pipelines for visual analysis of different sports using Computer Vision and Deep Learning.

### Link to slack workspace (Channel name :- # sports_analytics_cv) : https://join.slack.com/t/saidlseasonofcode/shared_invite/zt-f1oxhdtz-0cfoBOy15vbUNIL~NKPcDw

## To setup project, open up a new terminal and enter the following:
```
sh setup_conda.sh
```
OR
```
sh setup_pip.sh
```

## To test, run test.py or simply open up a new terminal and enter the following code: 
```
# For single image:

from Badminton.Badminton import Detector
obj = Detector()
obj.detect_players_image("Badminton/images/bad.jpg")

################################################################

# For video:

from Badminton.Badminton import Detector
obj = Detector()
obj.detect_players_video("Badminton/images/video2.mp4")

################################################################

# For using tiny yolo for better FPS:
# For video:

from Badminton.Badminton import Detector
obj = Detector(tiny=True)
obj.detect_players_video("Badminton/images/video2.mp4")

################################################################

# For Heatmap generation:

from Badminton.Badminton import Detector
obj = Detector()
obj.get_heatmap("Badminton/images/video2.mp4")

################################################################

```

## To test this code in Windows simply change the code in test.py to

```

from Badminton.Badminton import Detector
obj = Detector(Windows=True)
obj.detect_players_video("Badminton/images/video2.mp4")

```
