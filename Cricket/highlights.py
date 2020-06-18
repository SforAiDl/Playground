import librosa
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip 
import os

class Highlights:
  
  def __init__(self, path):
    if not os.path.isfile(path):
        raise FileNotFoundError("No file found at %s" %path)
    self.samples, self.sr = librosa.load(path,sr = 16000)                    # Load the amplitude samples and the sampling rate
    self.max_slice = 5                                                       # 5 samples per window
    self.window_length = self.max_slice * self.sr
    self.example = []
    self.example = np.array(self.example)
    self.energy = []
    self.energy = np.array(self.energy)
    self.thresh = 0
    self.flag = 0

  def energy_plot(self):        
    # plots the amplitude vs sample_number for a random segment of the audio file
    self.example = self.samples[21*self.window_length:22*self.window_length]
    print("Energy plot:\n")                                               
    self.energy = sum(abs(self.example**2))
    fig = plt.figure(figsize=(14, 8)) 
    y = fig.add_subplot(211) 
    y.set_xlabel('time') 
    y.set_ylabel('Amplitude') 
    y.plot(self.example)

  def distribution_plot(self): 
    print("Short time distribution plot:\n")  
    # Plots the short time distribution of energy (number of samples in an energy range)                                               
    self.energy = np.array([sum(abs(self.samples[i : i + self.window_length]**2)) for i in range(0, len(self.samples), self.window_length)])
    plt.hist(self.energy)
    plt.show()

  def set_threshold(self, threshold):
    # Sets the threshold amplitude above which the window will considered as eligible for the highlights
    #User can select the threshold based on this plot
    self.thresh = threshold
    self.flag = 1
    

  def generate(self, path): 
    # Generates the highlights. The selected chunks get stored in the current directory which can later be merged using a video editing software
    #path is the path to the video file corresponding to the wav file
    if (self.flag == 0):
      #energy_plot()
      self.distribution_plot()
      th = float(input(print("Enter the threshold value:")))
      print("\n")
      self.set_threshold(th)

    df=pd.DataFrame(columns=['energy','start','end'])
    row_index=0
    for i in range(len(self.energy)):
      value = self.energy[i]
      if(value >= self.thresh):
        i=np.where(self.energy == value)[0]
        df.loc[row_index,'energy'] = value
        df.loc[row_index,'start'] = i[0] * 5
        df.loc[row_index,'end'] = (i[0] + 1) * 5
        row_index= row_index + 1

    temp = []
    i = 0
    j = 0
    n = len(df) - 2
    m = len(df) - 1
    while(i <= n):
      j=i + 1
      while(j <= m):
        if(df['end'][i] == df['start'][j]):
          df.loc[i,'end'] = df.loc[j,'end']
          temp.append(j)
          j=j + 1
        else:
          i = j
          break  
    df.drop(temp, axis = 0,inplace = True)

    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    start=np.array(df['start'])
    end=np.array(df['end'])
    for i in range(len(df)):
      if(i!=0):
        start_lim = start[i] - 5
      else:
        start_lim = start[i] 
      end_lim   = end[i]   
      filename = "highlight" + str(i+1) + ".mp4"
      ffmpeg_extract_subclip(path, start_lim, end_lim, targetname = filename)