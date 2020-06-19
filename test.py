from Badminton import Detector

obj = Detector(tiny=True)
obj.detect_players_video("Badminton/images/video2.mp4",optimization = True, frames_skipped_input=5)