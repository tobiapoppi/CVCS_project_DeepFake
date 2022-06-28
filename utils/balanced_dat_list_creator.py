import os
import argparse
from pathlib import Path
import random
import glob
import shutil

def get_args():
    parser = argparse.ArgumentParser("Dataset Creator for DeepFake Detection Task")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Absolute path the root directory of the single sets.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Name the output file",
    )
    args = parser.parse_args()
    return args


def main(opt):
    output_file = opt.output
    main_path = Path(opt.path)

    imgs = sorted(sorted(main_path.rglob('*.png')) + sorted(main_path.rglob('*.jpg')))
    random.shuffle(imgs)

    print('There are ' + str(len(imgs)) + ' images.\n')
    print('Creating the file...')

    zero = 0
    one = 0
    for im in imgs:
        txt_file = os.path.join(im.parent, str(im.stem) + '.txt')
        with open(txt_file, 'r') as l:
            label = l.read()
            zero += 1-int(label)
            one += int(label)
    class_size = min(zero, one)


    with open(output_file, 'w') as f:
        zero = 0
        one = 0
        for im in imgs:
            txt_file = os.path.join(im.parent, str(im.stem) + '.txt')
            with open(txt_file, 'r') as l:
                label = l.read()
                if label==0:
                    print('uno zero ci è !!')
                zero += 1 - int(label)
                one += int(label)
            if zero<class_size and int(label)==0:
                f.write(str(im) + " " + str(label) + "\n")
            if one<class_size and int(label)==1:
                f.write(str(im) + " " + str(label) + "\n")
    print('Finished! Balanced data : {}'.format(class_size*2))


if __name__ == "__main__":
    options = get_args()
    main(options)