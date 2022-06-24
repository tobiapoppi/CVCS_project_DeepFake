# CVCS project DeepFake
Project for the exam "Computer Vision and Cognitive Systems"

This is a project for DeepFakes Detection, and we implemented several computer vision techniques in order to do this task.

We are using two datasets. 
First Dataset: https://iplab.dmi.unict.it/deepfakechallenge/#[object%20Object]

The dataset we first try to use is full training set, task 1.
Task 1 of the challenge in the link is the Detection task.

Second Dataset: https://github.com/ondyari/FaceForensics

## Dataset Creation

1. First download from the upper link files "0-CelebA.zip, 0-FFHQ.zip, 1-ATTGAN.zip, 1-GDWCT.zip, 1-StarGAN.zip, 1-STYLEGAN.zip, 1-STYLEGAN2.zip" from the section "release of full training set".

2. Extract all archives in a directory which I will call <sets_path>

3. With the script furnished by the FaceForensics repository, download all the subsets of FF++.

4. Run the script `dataset_creator.py` giving as parameters the root path of extracted sets of the first dataset (challenge), the root path of downloaded subsets of the second dataset (FF++), the dataset creation path (a new output folder) and, optionally, the split percentages of validation and test set. This script will automatically create the annotation .txt files with a 0 if the class is Real or 1 if the class is Fake.
* `python dataset_creator.py -c <sets_path_challenge_dataset> -f <ff++_dataset_main_folder> -o <output_path>/dataset`


5. If you also need the txt_list files (train.txt, val.txt and test.txt) containing the list of image paths followed by the label, you can use another script aswell.
* `python data_list_creator.py -p <train_set_path> -o <output_path>/train.txt`
