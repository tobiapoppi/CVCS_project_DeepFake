# CVCS_project_DeepFake
Project for the exam "Computer Vision and Cognitive Systems"


Dataset: https://iplab.dmi.unict.it/deepfakechallenge/#[object%20Object]

The dataset we first try to use is full training set, task 1.
Task 1 of the challenge in the link is the Detection task.


## Dataset Creation

1. First download from the upper link files "0-CelebA.zip, 0-FFHQ.zip, 1-ATTGAN.zip, 1-GDWCT.zip, 1-StarGAN.zip, 1-STYLEGAN.zip, 1-STYLEGAN2.zip" from the section "release of full training set".

2. Extract all archives in a directory which I will call <sets_path>

3. Run the script `dataset_creator.py` giving as parameters the root path of extracted sets, the dataset creation path (a new output folder) and optionally, the split percentages of validation and test set.
* `python dataset_creator.py -p <sets_path> -o <output_path>/dataset`
