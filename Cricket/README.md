# Cricket

## Generating Highlights

Usage : _from Cricket.highlights import Highlights_  
	mygame = Highlights(path) ; path - path to the audio file of the game

### Dependencies : 

* librosa
* matplotlib
* numpy
* IPython
* pandas
* moviepy
* os

### Description

Highlights is a class which contains a set of functions which will help you in generating highlights from a match video.

* Highlights.energy_plot() : Plots the energy distribution of samples taken at a rate of 16000Hz  from a segment of the audio file provided

* Highlights.distribution_plot() : Plots the short time distribution of energy (number of samples in an energy range). Can be used to the set the threshold above which the sample will be considered as a highlight segment.

* Highlights.set_threshold(threshold) : Sets the threshold above which a segment is considered as a highlight segment to the value provided as the argument

* Highlights.generate(path_to_video file) : Generates non-coherent highlight segments and stores them in the current directory.
