# CVCS project DeepFake
Project for the exam "Computer Vision and Cognitive Systems"

This is a project for DeepFakes Detection, and we implemented several computer vision techniques in order to do this task.

We are using two datasets. 
First Dataset: https://iplab.dmi.unict.it/deepfakechallenge/#[object%20Object]

The dataset we first try to use is full training set, task 1.
Task 1 of the challenge in the link is the Detection task.

Second Dataset: https://github.com/ondyari/FaceForensics

## Setting up the environment
* `conda create -n cvcs python=3.7`
* `conda activate cvcs`
* `pip install -r requirements.txt`


## Dataset Creation

1. First download from the upper link files "0-CelebA.zip, 0-FFHQ.zip, 1-ATTGAN.zip, 1-GDWCT.zip, 1-StarGAN.zip, 1-STYLEGAN.zip, 1-STYLEGAN2.zip" from the section "release of full training set".

2. Extract all archives in a directory which I will call `<sets_path>`

3. With the script furnished by the FaceForensics repository, download all the subsets of FF++.

4. Now you have to extract the frames from the downloaded ff++ sequences, because they are videos.
* `python utils/get_frames.py -i <ff++_dataset_path> -o <output_folder_path>` You can also set the number of frames you want to extract from each video with the parameter `-f`

5. Run the script `dataset_creator.py` giving as parameters the root path of extracted sets of the first dataset (challenge), the root path of downloaded subsets of the second dataset (FF++), the dataset creation path (a new output folder) and, optionally, the split percentages of validation and test set (`-sv` and `-st`). This script will automatically create the annotation .txt files with a 0 if the class is Real or 1 if the class is Fake.
* `python utils/dataset_creator.py -c <sets_path_challenge_dataset> -f <ff++_dataset_main_folder> -o <output_path>/dataset`


6. If you also need the txt_list files (train.txt, val.txt and test.txt) containing the list of image paths followed by the label, you can use another script aswell.
* `python utils/data_list_creator.py -p <train_set_path> -o <output_path>/train.txt`

## Dataset for Head Detection
The dataset we used to train the head-detector is "HollywoodHeads", which is presented in this paper: https://arxiv.org/pdf/1511.07917.pdf .
This dataset is in the Pascal-VOC format.

* Download the dataset: `wget -P data http://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip`
* Unpack it: `unzip data/HollywoodHeads.zip -d data`
* Remove all the images and annotations without a head: `python dataset_register.py` (Rember to specify your dataset path inside the script's main!)

## EfficientDet Head Detector Training

First, we did a training starting from imagenet weights, freezing backbone layers.
* `python train.py --snapshot imagenet --phi 0 --gpu 0 -- random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 1000 --pascal <hollywoodheads_dataset_path>`

Then, a second training, starting from the weights of last epoch of first training, freezing BatchNormalization layers, and using a higher steps value.
Increasing the steps means forcing the CNN to process a lot more data in a single epoch. In addition, we reduced batch size.
* `python train.py --snapshot <path_to_model.h5> --phi 0 --gpu 0 --weighted-bifpn --random-transform --compute-val-loss --freeze-bn --batch-size 4 --steps 10000 pascal /data/HollywoodHeads/`

Now we can select, in the 50 epochs weights of last training, the one with the best trade-off in the train-val loss graph.

## Inference of EfficientDet on CVCS Dataset
Now we need to inference efficientDet model on our dataset, in order to create a new copy of CVCS dataset, but only with cropped heads. We will use this dataset to train xception net.
Then, for DeepFake detection inference, we provide a script which first detect the head and then classify it fake or not. So the final system will also generalize on non-cropped images.

* `cd CVCS_project_DeepFake`
* `python -m CVCS_project_DeepFake.efficientDetheaddetection.inference`

n.b.: be sure to set all the correct paths in `inference.py`


