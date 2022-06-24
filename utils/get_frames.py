from ast import Break, arg
import cv2
import os
import argparse
import subprocess

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_args():
    parser = argparse.ArgumentParser("Dataset Creator for DeepFake Detection Task")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="Absolute path of the directory where you said FaceForensics to save the videos",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default='frames',
        help="Name the output directory where you want to save the frames",
    )
    parser.add_argument(
        "-f",
        "--frames",
        default=3,
        type=int,
        help="Number of frame you want to extract from each video",
    )
    args = parser.parse_args()
    return args

args = get_args()

if args.input_path == None:
    print('Please, specify "--input_path", the absolute path of the directory where you said FaceForensics to save the videos')
    exit()

input_path = args.input_path

if 'manipulated_sequences' not in os.listdir(input_path):
    print('It seems there is no "manipulated_sequences" subdirectory in the input_directory you specified :/')
    exit()

if args.output_directory not in os.listdir('.'):
    os.mkdir('./'+args.output_directory)

num_frames = args.frames
print('Start Processing Manipulated Sequences!\n')
for dir in os.listdir(f'{input_path}/manipulated_sequences/'):
    print(f'{dir}\n')
    count_vid = 0
    for quality in os.listdir(f'{input_path}/manipulated_sequences/{dir}'):
        for video in os.listdir(f'{input_path}/manipulated_sequences/{dir}/{quality}/videos/'):
            last = -1
            path = f'{input_path}/manipulated_sequences/{dir}/{quality}/videos/{video}'
            vidcap = cv2.VideoCapture(path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            durata = get_length(path)
            tot_frames = durata * fps
            step = tot_frames / num_frames
            success,image = vidcap.read()
            count_frame = 0
            for i in range(num_frames):
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, i*step)
                success,image = vidcap.read()
                if success:
                    count_frame += 1
                    cv2.imwrite(f"./{args.output_directory}/manipulated_{dir}_{quality}_{video[:-4]}__{count_frame}.jpg", image)     # save frame as JPEG file
                else:
                    print('error!')

            count_vid += 1
            if not count_vid%100:
                print(f'{count_vid} video processed...')
print('Finished Original Sequences Processing!')
        
###############################################
###      PROCESSING ORIGINAL SEQUENCES      ###
###############################################

#original sequences
print('Start Processing Original Sequences!\n')
count_vid = 0
for video in os.listdir(f'{input_path}/original_sequences/c23/videos/'):
    last = -1
    path = f'{input_path}/original_sequences/c23/videos/{video}'
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    durata = get_length(path)
    tot_frames = durata * fps
    step = tot_frames / num_frames
    success,image = vidcap.read()
    count_frame = 0
    
    for i in range(num_frames):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i*step)
        success,image = vidcap.read()
        if success:
            count_frame += 1
            cv2.imwrite(f"./{args.output_directory}/original_{quality}_{video[:-4]}__{count_frame}.jpg", image)     # save frame as JPEG file  
        else:
            print('error!')
    
    count_vid += 1
    if not count_vid%100:
        print(f'{count_vid} video processed...')
print('Finished Original Sequences Processing!')

#deepFakeDetection orignial sequences
print('Start Processing Original Sequences (DeepFakeDetection)!\n')
count_vid = 0
for video in os.listdir(f'{input_path}/original_sequences/actors/c23/videos/'):
    last = -1
    path = f'{input_path}/original_sequences/actors/c23/videos/{video}'
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    durata = get_length(path)
    tot_frames = durata * fps
    step = tot_frames / num_frames
    success,image = vidcap.read()
    count_frame = 0
    
    for i in range(num_frames):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i*step)
        success,image = vidcap.read()
        if success:
            count_frame += 1
            cv2.imwrite(f"./{args.output_directory}/original_DFD_{quality}_{video[:-4]}__{count_frame}.jpg", image)     # save frame as JPEG file  
        else:
            print('error!')
    
    count_vid += 1
    if not count_vid%100:
        print(f'{count_vid} video processed...')

print('Finished Original Sequences Processing (DeepFakeDetection)!')
 
            # while success:
            #     success,image = vidcap.read()
            #     if success:
            #         count += 1
            #     if last == -1 or (count-last) >= step:
            #         last = count
            #         cv2.imwrite(f"./{args.output_directory}/original_{quality}_{video[:-4]}__{count}.jpg", image)     # save frame as JPEG file  
            # print(f'frames: {count}, durata: {durata}, fps calcolati: {count/durata}')
        