# Color-Clustering-Via-GMM-EM
# Overview
This project involves detecting buoys in videos using Gaussian Mixture Models (GMM) and Gaussian techniques. It includes several Python scripts for training and detection of buoys of different colors using 1D and 2D Gaussian models.
# Directory Structure 

# Training images folders:
1.	"green_train" - Contains training images for detecting green buoys.
2.	"yellow_train" - Contains training images for detecting yellow buoys.
3.	"orange_train" - Contains training images for detecting orange buoys.


# Python scripts:
1.	takeimage.py : Code to cut out buoys from frames to generate the training set.
2.	1D_gauss.py : Code to generate random Gaussian data samples and find means and standard deviations using the Expectation-Maximization (EM) algorithm.
3.	1D_gauss_green.py : Code to detect green buoys using a 1D Gaussian model on the green channel.
4.	1D_gauss_yellow.py : Code to detect yellow buoys using a 1D Gaussian model on the green and red channels.
5.	1D_gauss_orange.py : Code to detect orange buoys using a 1D Gaussian model on the red channel.
6.	2D_gauss_green.py : Code to detect green buoys using a 2D Gaussian model on all RGB channels.
7.	2D_gauss_yellow.py : Code to detect yellow buoys using a 2D Gaussian model on all RGB channels.
8.	2D_gauss_orange.py : Code to detect orange buoys using a 2D Gaussian model on all RGB channels.
9.	2D_gauss_all.py : Code to detect all buoys using a 2D Gaussian model on all RGB channels.
    
# Videos included:
1.	detectbuoy.avi :The video we work with
2.  1D_gauss_green.avi : Output of the 1D_gauss_green.py script.
3.	1D_gauss_yellow.avi : Output of the 1D_gauss_yellow.py script.
4.	1D_gauss_orange.avi : Output of the 1D_gauss_orange.py script.
5.  2D_gauss_green.avi : Output of the 2D_gauss_green.py script.
6.	2D_gauss_yellow.avi : Output of the 2D_gauss_yellow.py script.
7.	2D_gauss_orange.avi : Output of the 2D_gauss_orange.py script.
8.	2D_gauss_all.avi : Output of the 2D_gauss_all.py script.

The python files must be kept in the same folder as the folders containing the training images. Each file can then be run from the command line.

## Instructions

# Setup:
Ensure that the Python files and folders containing the training images are in the same directory.
# Training:
Run takeimage.py to cut out buoys from video frames and generate the training set images.
# Gaussian Modeling:
1. Use 1D_gauss.py to generate Gaussian data samples and find the means and standard deviations for the 1D Gaussian models.
2. Run 1D_gauss_green.py, 1D_gauss_yellow.py, and 1D_gauss_orange.py to detect buoys of respective colors using 1D Gaussian models.
# 2D Gaussian Detection:
Use 2D_gauss_green.py, 2D_gauss_yellow.py, 2D_gauss_orange.py, and 2D_gauss_all.py to detect buoys of respective colors using 2D Gaussian models.
# Testing:
The detectbuoy.avi video can be processed with the above scripts to generate output videos.
# Running the Scripts:
Each Python file can be run from the command line using the following format:
python <script_name>.py
