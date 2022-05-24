# Car-Crash-Prediction-C3D
Car crash prediction using the C3D 3D-CNN architecture 

# Steps to run the code

Install Anaconda 

Clone this repo

Navigate to the local directory of this repo through terminal 

Open terminal and run : conda env create -f car_crash_pred_env.yml (this command will create a conda environment with all the required libraries installed) 

You can see the newly created environment using this command: conda list env

The newly created environment should be called Crash_pred 

To activate the environment, use this command: conda activate Crash_pred 

Download the dataset into the directory where the scripts are stored using this specific link: https://drive.google.com/file/d/1l-Ps5LkRqt_xVJHivaU8ZI216ySh4zBa/view?usp=sharing (the videos here are labeled according to their class) 

Update the video directory path in the crash_pred_C3D.py script to reflect the location where you saved the data. 

Change the parameters skip, img_cols, img_rows, depth according to your needs. 

To begin training, in terminal run this command: python crash_pred_C3C.py  
