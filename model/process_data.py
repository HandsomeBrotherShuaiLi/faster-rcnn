import os,cv2,numpy as np,random
import xml.etree.ElementTree as ET
from collections import defaultdict
def get_data(input_path):
    all_imgs=[]
    classes_count=defaultdict(int)
    class_mapping={}
    data_paths=input_path
    print("start to process annotation file")
    annotation_path=input_path+'/annotation'
    imgs_path=input_path+'/image/'
    idx=0
    for annotation_file in os.listdir(annotation_path):
        try:
            idx+=1
            et=ET.parse(input_path+'/annotation/'+annotation_file)
            element=et.getroot()
            element_objects=element.findall('object')
            element_filename=element.find('filename').text
            element_width=int(element.find('size').find('width').text)
            element_height=int(element.find('size').find('height').text)
            if len(element_objects)>0:
                annotation_data={'filepath':imgs_path+element_filename,'width':element_width,
                                 'height':element_height,'bboxes':[],'type':"train"}
            for element_obj in element_objects:
                class_name=element_obj.find('name').text
                if class_name not in classes_count:
                    classes_count[class_name]=1
                else:
                    classes_count[class_name]+=1
                if class_name not in class_mapping:
                    class_mapping[class_name]=len(class_mapping)
                obj_bbox=element_obj.find('bndbox')
                x1=int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty=int(element_obj.find('difficult').text)==1
                annotation_data['bboxes'].append(
                    {'class':class_name,'x1':x1,'x2':x2,'y1':y1,'y2':y2,
                     'difficult':difficulty}
                )
            all_imgs.append(annotation_data)
        except Exception as e:
            print(e)
            continue
    random.shuffle(all_imgs)
    for i in range(len(all_imgs)):
        if i>=98 and i <140:
            all_imgs[i]['type']='val'
        elif i>=140:
            all_imgs[i]['type']='test'
        else:
            pass
    return all_imgs,classes_count,class_mapping