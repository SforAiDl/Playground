# Playground
A python library consisting of pipelines for visual analysis of different sports using Computer Vision and Deep Learning.

### Update : Badminton detector has been added

## Usage:
```
from Badminton.Badminton import Detector
detector_object = Detector()
detector_object.detect_players('/Badminton/images/bad.jpg')
```

Note: Weights can be downlaoded using the download_weights.sh script in Badminton/config

## Sample Output of Detector

![Example of output of Detector](./Badminton/sample_output.png)
