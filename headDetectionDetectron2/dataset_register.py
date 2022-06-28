from pathlib import Path
import os
import cv2
import xml.etree.ElementTree as ET
import xmltodict

def hollywood_heads_dataset(hollywood_dataset_path):

    my_hollywood_dataset_path = Path(hollywood_dataset_path)
    img_path = Path(os.path.join(my_hollywood_dataset_path, 'JPEGImages'))
    ann_path = Path(os.path.join(my_hollywood_dataset_path, 'Annotations'))
    splits_path = Path(os.path.join(my_hollywood_dataset_path, 'Splits'))

    imgs = sorted(img_path.rglob('*.jpeg'))
    print(f'There are {len(imgs)} images.')
    custom_dict = []

    id = 0

    bad_imgs = []

    for im in imgs:
        file_name = str(im)
        annotations = []
        label_path = os.path.join(ann_path, (str(im.stem)+'.xml'))
        
        tree = ET.parse(label_path)
        xml_data = tree.getroot()
        xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')
        data_dict = dict(xmltodict.parse(xmlstr))
        bbox_mode = 0
        category_id = 0 #always 0, we need only the faces
        height = data_dict['annotation']['size']['height']
        width = data_dict['annotation']['size']['width']
        if type(data_dict['annotation']['object']) == list:
            for head in data_dict['annotation']['object']:
                xmin = head['bndbox']['xmin']
                ymin = head['bndbox']['ymin']
                xmax = head['bndbox']['xmax']
                ymax = head['bndbox']['ymax']
                bbox = [xmin, ymin, xmax, ymax]
                annotations.append({'bbox':bbox, 'bbox_mode':bbox_mode, 'category_id':category_id, 'segmentation':[], 'keypoints':[]})
        elif type(data_dict['annotation']['object']) == dict:
            xmin = data_dict['annotation']['object']['bndbox']['xmin']
            ymin = data_dict['annotation']['object']['bndbox']['ymin']
            xmax = data_dict['annotation']['object']['bndbox']['xmax']
            ymax = data_dict['annotation']['object']['bndbox']['ymax']
            bbox = [xmin, ymin, xmax, ymax]
            annotations.append({'bbox':bbox, 'bbox_mode':bbox_mode, 'category_id':category_id, 'segmentation':[], 'keypoints':[]})
        else:
            print("I am deleting img", str(im), " and label ", str(label_path))
            bad_imgs.append(str(im.stem))
            
            os.remove(im)
            os.remove(label_path)
            pass #if no labels are in the image, annotation is = []

        custom_dict.append({'file_name':file_name, 'height':height, 'width':width, 'image_id':str(id), 'annotations':annotations})
        
        if not id%1000:
            print(f'{id} images processed.')
        id += 1
    
    print("Cleaning .txt list files")
    with open(str(os.path.join(splits_path, 'test.txt'))) as oldfile, open(str(os.path.join(splits_path, 'test_new.txt')), 'w') as newfile:
        for line in oldfile:
            if not any(bad_img in line for bad_img in bad_imgs):
                newfile.write(line)
    with open(str(os.path.join(splits_path, 'val.txt'))) as oldfile, open(str(os.path.join(splits_path, 'val_new.txt')), 'w') as newfile:
        for line in oldfile:
            if not any(bad_img in line for bad_img in bad_imgs):
                newfile.write(line)
    with open(str(os.path.join(splits_path, 'train.txt'))) as oldfile, open(str(os.path.join(splits_path, 'train_new.txt')), 'w') as newfile:
        for line in oldfile:
            if not any(bad_img in line for bad_img in bad_imgs):
                newfile.write(line)
    
    os.remove(str(os.path.join(splits_path, 'test.txt')))
    os.remove(str(os.path.join(splits_path, 'train.txt')))
    os.remove(str(os.path.join(splits_path, 'val.txt')))

    os.rename(str(os.path.join(splits_path, 'test_new.txt')), str(os.path.join(splits_path, 'test.txt')))
    os.rename(str(os.path.join(splits_path, 'train_new.txt')), str(os.path.join(splits_path, 'train.txt')))
    os.rename(str(os.path.join(splits_path, 'val_new.txt')), str(os.path.join(splits_path, 'val.txt')))
    

    return custom_dict



if __name__ == '__main__':
    dataset_path = '/home/tobi/cvcs/data/HollywoodHeads/'
    hollywood_heads_dataset(dataset_path)