from Badminton import Detector

obj = Detector(tiny=True)
obj.get_heatmap("Badminton/images/video2.mp4",optimization = True)