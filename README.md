Overview

This project involves detecting buoys in videos using Gaussian Mixture Models (GMM) and Gaussian techniques. It includes several Python scripts for training and detection of buoys of different colors using 1D and 2D Gaussian models.

Directory Structure

Training Images Folders:
green_train/ - Contains training images for detecting green buoys.
yellow_train/ - Contains training images for detecting yellow buoys.
orange_train/ - Contains training images for detecting orange buoys.
Python Scripts:
takeimage.py - Code to cut out buoys from frames to generate the training set.
1D_gauss.py - Code to generate random Gaussian data samples and find means and standard deviations using the Expectation-Maximization (EM) algorithm.
1D_gauss_green.py - Code to detect green buoys using a 1D Gaussian model on the green channel.
1D_gauss_yellow.py - Code to detect yellow buoys using a 1D Gaussian model on the green and red channels.
1D_gauss_orange.py - Code to detect orange buoys using a 1D Gaussian model on the red channel.
2D_gauss_green.py - Code to detect green buoys using a 2D Gaussian model on all RGB channels.
2D_gauss_yellow.py - Code to detect yellow buoys using a 2D Gaussian model on all RGB channels.
2D_gauss_orange.py - Code to detect orange buoys using a 2D Gaussian model on all RGB channels.
2D_gauss_all.py - Code to detect all buoys using a 2D Gaussian model on all RGB channels.
Videos:
detectbuoy.avi - The video used for testing.
1D_gauss_green.avi - Output of the 1D_gauss_green.py script.
1D_gauss_yellow.avi - Output of the 1D_gauss_yellow.py script.
1D_gauss_orange.avi - Output of the 1D_gauss_orange.py script.
2D_gauss_green.avi - Output of the 2D_gauss_green.py script.
2D_gauss_yellow.avi - Output of the 2D_gauss_yellow.py script.
2D_gauss_orange.avi - Output of the 2D_gauss_orange.py script.
2D_gauss_all.avi - Output of the 2D_gauss_all.py script.
Instructions

Setup:
Ensure that the Python files and folders containing the training images are in the same directory.
Training:
Run takeimage.py to cut out buoys from video frames and generate the training set images.
Gaussian Modeling:
Use 1D_gauss.py to generate Gaussian data samples and find the means and standard deviations for the 1D Gaussian models.
Run 1D_gauss_green.py, 1D_gauss_yellow.py, and 1D_gauss_orange.py to detect buoys of respective colors using 1D Gaussian models.
2D Gaussian Detection:
Use 2D_gauss_green.py, 2D_gauss_yellow.py, 2D_gauss_orange.py, and 2D_gauss_all.py to detect buoys of respective colors using 2D Gaussian models.
Testing:
The detectbuoy.avi video can be processed with the above scripts to generate output videos.
Running the Scripts:
Each Python file can be run from the command line using the following format:
python <script_name>.py
Replace <script_name> with the appropriate file name.
