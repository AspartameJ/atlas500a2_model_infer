"""
-*- coding:utf-8 -*-
"""
import sys
import os
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "../acllite/"))
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
import constants as const
import cv2
#set path 
currentPath =  os.path.join(path)
OUTPUT_DIR = os.path.join(currentPath, 'out')
MODEL_PATH = os.path.join(currentPath,"resnet50.om")

#set model size 
DEVICE_ID=0


class Print_result():
    def __init__(self):
        self.class_name = ['AgeLess18', 'Age1860', 'AgeOver60', 'Female', 'Front', 'Side', 'Back']
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def decode(self, valid_logits):
        valid_probs = self.sigmoid(valid_logits)
        age = self.class_name[valid_probs[0].tolist().index(max(valid_probs[0][0:2], key=abs))]
        gender = 'Male' if valid_probs[0][3] < 0.96 else 'Female'
        side = self.class_name[valid_probs[0].tolist().index(max(valid_probs[0][4:6]))]
        return age, gender, side


Print_result = Print_result()

class Classify(object):
    """
    Class for portrait segmentation
    """
    def __init__(self, model_path):
        self._model_path = model_path
        self.acl_resource = AclLiteResource(device_id=DEVICE_ID)
        self.acl_resource.init()
        self._model = AclLiteModel(self._model_path)

    def pre_process(self, input_path):
        """
        preprocess 
        """
        with Image.open(input_path) as img:
            img = np.array(img.convert('RGB'))
        img = img / 255.
        img = cv2.resize(img, (192, 256))
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.transpose(img, axes=[2,0,1])
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img


    def inference(self, input_data):
        """
        model inference
        """
        return self._model.execute(input_data)


    def post_process(self,Print_result,valid_logits):
        """
        Post-processing, analysis of inference results
        """
        age, gender, side = Print_result.decode(valid_logits)
        print(age, gender)


def main():
    image_dir = os.path.join("testpic" )
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    classify = Classify(MODEL_PATH)
    images_list = [os.path.join(image_dir, img)
                   for img in os.listdir(image_dir)
                   if os.path.splitext(img)[1] in const.IMG_EXT]

    for image_file in images_list:
        print('=== ' + os.path.basename(image_file) + '===')
        resized_image = classify.pre_process(image_file)
        valid_logits, attns = classify.inference(resized_image)
        classify.post_process(Print_result,valid_logits)


if __name__ == '__main__':
    main()
