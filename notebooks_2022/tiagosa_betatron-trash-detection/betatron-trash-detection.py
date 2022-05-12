#!/usr/bin/env python
# coding: utf-8

# # Trash Detection
# 
# ### Equipe composta por:
#     - Eduardo Sales
#     - Isaque Vilson
#     - Tiago Sá
#     
# #### Objetivo Geral
# 
# Assim, implementar um modelo basedo em Aprendizado Profundo para detectar diferentes lixeiras cheias e vazias nas ruas de Manaus.
# 
# #### Objetivos Específicos
# 
# 1. Implementação de um modelo a partir da abordagem "Transfer Learning" com YOLOv5;
# 2. Aplicar métodos de data augmentation para aumentar a quantidade de imagens no conjunto de treino;
# 3. Comparação entre os modelos em termos de métricas de validação (acurácia, precisão, revocação e f1-score).
# 
# ### Topics:
# 
# #### Visão Geral
# 
# 1. Sobre o Dataset;
# 
# 2. Análise Exploratória
# 
# 
# #### Divisão do dataset
#     - Treino e validação
# 
# #### Pipeline 1 - Transfer Learning
# 1. Preprocessamento;
# 
#     1.1 No data augmentation
#     
# 2. Treinamento;
# 
#     
# 3. Avaliação;
# 
#     3.1 Precisão, Revocação, Matriz de Confusão, etc...
#     
# 
# #### Pipeline 2 - Data augmentation
# 
# 1. Preprocessamento;
# 
#     1.1 Técnicas comuns de data augmentation
#     
# 2. Treinamento;
# 
#     
# 3. Avaliação
# 
#     3.1 Precisão, Revocação, Matriz de Confusão...
#     
#    
# #### Comparação entre os resultados obtidos a partir da pipeline 1 e 2
# 
# 
# ####  Considerações Finais
# 
# 
# 
# 
# 
# #### Data: 05/05/2022

# # Download YOLOv5

# In[ ]:


get_ipython().system('python --version')


# In[ ]:


def list_dir(path):
    
    """ List everything inside the directory """
    filenames = os.listdir(path)
    
    # return a list of filenames
    
    return [filename for filename in filenames]


# In[ ]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone')


# In[ ]:


get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().run_line_magic('pip', 'install -qr requirements.txt # install')

import torch

from yolov5 import utils
display = utils.notebook_init() # check


# In[ ]:


import cv2 # Computacional Vision manipulations on Images from Taco Dataset
import albumentations as A # To pre processing images and also bounding boxes parameters

import numpy as np # To perform numeric operations on Digital Image Processing
import pandas as pd # To manipulate dataframes
import matplotlib.pyplot as plt # To visualize images
import seaborn as sns
import random # To visualize and image
from IPython.display import Image, display

import glob
import os
import shutil
from sklearn.model_selection import train_test_split


# # Exploratory Data Analysis

# In[ ]:


# Bounding Boxes Colors

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    
    """ Visualizes a single bounding boxes on the image """
    
    
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color = color, thickness = thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    


# In[ ]:


def convert_yolo_to_coco(image, bboxes, category_ids, category_id_name):
    
    height, width, _ = image.shape
    
    coco_bboxes = []
    
    for bbox in bboxes:
        alb_bbox = A.bbox_utils.convert_bbox_to_albumentations(bbox,
                                                                'yolo',
                                                                height,
                                                                width,
                                                                check_validity=False)
    
        coco_bbox = A.bbox_utils.convert_bbox_from_albumentations (alb_bbox,
                                                                   'coco',
                                                                   height,
                                                                   width,
                                                                   check_validity=False)
        coco_bboxes.append(coco_bbox)
    return coco_bboxes


# In[ ]:


img_path = '/kaggle/input/trash-detection/train/images/eduardo14.png'


image               = cv2.imread(img_path)
bboxes              = [[0.427332, 0.569239, 0.062907, 0.153277],
                       [0.486714, 0.446089, 0.045011, 0.107822],
                       [0.767625, 0.315539, 0.026573, 0.028541]]
category_ids        = [0, 0, 0]
category_id_to_name = {0:"empty"}

bboxes = convert_yolo_to_coco(image, bboxes, category_ids, category_id_to_name)


# In[ ]:


# Visualize the random image with associated bounding boxes
visualize(image, bboxes, category_ids, category_id_to_name)


# # Split

# In[ ]:


def create_dir(path):
    """ Criar diretórios """
    if (os.path.exists(path) and (os.listdir(path) != [])):
        shutil.rmtree(path) 
        
    else:
        os.makedirs(path)
        
    


# In[ ]:


def get_imgs_paths(path):
    """ Lista todos os arquivos .png do diretório especificado """
    return glob.glob(os.path.join(path, '*.png'))


# In[ ]:


src_dir = '/kaggle/input/trash-detection/'
dest_dir = '/kaggle/working/trash-detection/'

def split_train_test(src_dir, dest_dir):
    
    # Setting train and test paths
    train_dir = os.path.join(dest_dir, 'train')
    test_dir  = os.path.join(dest_dir, 'val')
    
    # Creating directories
    create_dir(dest_dir)
    create_dir(os.path.join(train_dir, 'images'))
    create_dir(os.path.join(train_dir, 'labels'))      
    
    create_dir(os.path.join(test_dir, 'images'))
    create_dir(os.path.join(test_dir, 'labels'))              
               
    # Getting images paths
    imgs_paths = get_imgs_paths(os.path.join(src_dir, 'train', 'images'))
    
    # Creating dataframe with imgs paths
    df = pd.DataFrame({'imgs' : imgs_paths})
    
    # Train test split
    train, test = train_test_split(df, test_size = 0.2, random_state = 101)
    
    print(f" Train shape: {train.shape}\n")
    print(f" Test shape: {test.shape}\n")
    
    splits = [train, test]
    
    # Moving images.
    for split in splits:
        for img in split.imgs:
            filename_no_ext = os.path.splitext(os.path.basename(img))[0]
            if split is train:
                shutil.copy(img, os.path.join(train_dir, 'images', filename_no_ext + '.png'))
                shutil.copy(os.path.join(src_dir,'train', 'labels', filename_no_ext + '.txt'),
                           os.path.join(train_dir, 'labels', filename_no_ext + '.txt'))
    
            else:
                shutil.copy(img, os.path.join(test_dir, 'images', filename_no_ext + '.png'))
                shutil.copy(os.path.join(src_dir,'train', 'labels', filename_no_ext + '.txt'),
                           os.path.join(test_dir, 'labels', filename_no_ext + '.txt'))    


# In[ ]:


split_train_test(src_dir, dest_dir)


# In[ ]:


# Quantidade de imagens e labels.


# # Treinamento com YOLOv5

# In[ ]:


get_ipython().system("python train.py --img 712 --batch 64 --epochs 250 --data '/kaggle/input/trash-detection/data.yaml' --weights yolov5s.pt --cache")


# In[ ]:


get_ipython().system('ls /kaggle/working/yolov5/runs/train/exp')


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/results.png'))


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/F1_curve.png'))


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/PR_curve.png'))


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/confusion_matrix.png'))


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/train_batch1.jpg'))


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/labels.jpg'))


# In[ ]:





# # Data Augmentation Techniques
# 
# ## Foi utilizado o roboflow como ferramenta para data augmentation

# In[ ]:


get_ipython().system('pip install roboflow')

from roboflow import Roboflow
rf = Roboflow(api_key="oVr9WVxspeXzy9q3oEya")
project = rf.workspace("trash-detection-lqvbn").project("trash-detection-qcyf3")
dataset = project.version(2).download("yolov5")


# In[ ]:


get_ipython().system('ls Tras')


# In[ ]:


get_ipython().system("python train.py --img 712 --batch 64 --epochs 250 --data '/kaggle/working/yolov5/Trash-Detection-2/data.yaml' --weights yolov5s.pt --cache")


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/results.png'))


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/F1_curve.png'))


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/confusion_matrix.png'))


# In[ ]:


display(Image(filename = '/kaggle/working/yolov5/runs/train/exp/train_batch1.jpg'))


# In[ ]:





# # Conclusões

# ## Os resultados foram muito ruins. Os modelos treinados apresentaram um alto overfitting. Isso é possível verificar pois, durante o treinamento, o erro é minímo ao longo das últimas épocas, porém, ao verificar a performance no conjunto de validação, o erro aumenta drasticamente ao longo das épocas.
# 
# ## Nem mesmo aplicar técnicas de data augmentation provocaram melhorias significativas no modelo.
# 
# ## Dentre as possíveis ações para melhorar os resultados, destacam-se:
# 
# * Para contornar essa situação, mais imagens deveriam ser rotuladas. Ao todo, foram coletadas aproximadamente 220 imagens. Essa quantidade é muito pouco para a complexidade de objetos ao redor da lixeira. No geral, as imagens possuem misturaas de asfalto, casas, pessoas, animais, carros, e inúmeros outros objetos que dificultam o modelo a aprender quais são os traços da lixeira em um contexto tão complexo. Além disso, foram coletadas diferentes tipos de lixeira, porém poucos exemplos. Acrescenta-se que as imagens foram capturadas sob diferentes condições de iluminação, ângulo e posição das lixeiras, porém poucos exemplos. Em virtude das imagens serem coletadas pelo google maps, muitas vezes as lixeiras possuem um formato distorcido em função da lente esférica que é utilizada para capturar essas imagens.
# 
# * As imagens poderiam ser mais centralizadas nas lixeiras. Isso potencialmente tornaria mais fácil o modelo adaptar seus pesos para aprender os traços de lixeira. No dataset gerado, as imagens possuem exemplos de lixeiras que estão muito distantes, com uma área de cobertura de pixel muito inferior a dimensão original da imagem.

# In[ ]:




