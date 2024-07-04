# -*- coding: utf-8 -*-
import os
import json

def tojson(path):
    imagepath="train/images"
    labelpath="train/labels"
    images=os.listdir(path+imagepath)
    res={"training":[],"validation":[]}
    template={"image":"","label":""}
    for image in images:
        temp=template.copy()
        temp["image"]=path+imagepath+'/'+image
        temp["label"]=path+labelpath+'/'+image
        res["training"].append(temp)

    imagepath="val/images"
    labelpath="val/labels"
    images=os.listdir(path+imagepath)
    for image in images:
        temp=template.copy()
        temp["image"]=path+imagepath+'/'+image
        temp["label"]=path+labelpath+'/'+image
        res["validation"].append(temp)

    with open('datasets/dataset.json','w') as f:
        f.write(json.dumps(res))

