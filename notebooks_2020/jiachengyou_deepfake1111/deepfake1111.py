#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import matplotlib.patches as patches
import math
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
from blazeface import BlazeFace
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import PIL
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/kaggle/input/faceforensics'
os.listdir(path)
get_ipython().system('pip install --upgrade efficientnet-pytorch')
from efficientnet_pytorch import EfficientNet


# In[ ]:


def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]
        print(xmin,ymin)

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=detections[i, 16])
                ax.add_patch(circle)
        
    plt.show()

def findFace(img, detections):
#         print("detections nums", len(detections))
    i = 0
    ymin = math.floor(detections[i, 0] * img.shape[0])
    xmin = math.floor(detections[i, 1] * img.shape[1])
    ymax = math.ceil(detections[i, 2] * img.shape[0])
    xmax = math.ceil(detections[i, 3] * img.shape[1])
#         print(detections[i, 0], detections[i, 1],detections[i, 2], detections[i, 3])
#         print(img.shape)
#         print(xmin, xmax, ymin, ymax)
    face = img[ymin:ymax, xmin:xmax+1]
#         print(face.shape)

    return face


# In[ ]:


result = {
    "Deepfakes" : 429 / 500,    
    "Face2Face" : 276 / 500,
    "FaceSwap" : 287 / 500,
    "NeuralTextures" : 209 / 500,
    "origin" : 302 / 500 
}


# In[ ]:


class VideoProcess():
    
    def __init__(self):
        pass
    
    def getFrame(self, videos_dir, file_num="all", frame_num=5, type=0):
        
        file_list = os.listdir(videos_dir)
        file_list.sort()
        
        if len(file_list) == 0 : return None
        
        if file_num == 'all' :
            file_list = file_list[:]
        elif len(file_list) > file_num:
            file_list = file_list[:file_num]
        else:
            file_list = file_list[:]
            
        
        frame_num = 5
        pic_num = len(file_list)
        img_arr = [[] for i in range(pic_num)]
        for (index1,file) in enumerate(file_list):
            filename = os.path.join(videos_dir, file)
            cap=cv2.VideoCapture(filename)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0 : continue
            
            # type : linspace
            if type == 0:
                frame_idxs = np.linspace(0, frame_count - 1, frame_num, endpoint=True, dtype=np.int)
                frame_idxs = np.unique(frame_idxs)

            for (index2,idx) in enumerate(frame_idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame  = cap.read()
                img_arr[index1].append(frame)
            
            if index1 % 100 == 0:
                print("getFrame:", index1)
            
        return img_arr
    
        

    def blazeFace(self, imgs):

        gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = BlazeFace().to(gpu)
        net.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
        net.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")

        # Optionally change the thresholds:
        net.min_score_thresh = 0.75
        net.min_suppression_threshold = 0.3
        pic_num = len(imgs)

        if pic_num == 0: return None

        frame_num = len(imgs[0])

        res = [[] for i in range(pic_num)]
        for i in range(pic_num):
            for j in range(frame_num):
                img = imgs[i][j]
                detections = net.predict_on_image(cv2.resize(img,(128,128)))
                if len(detections) == 0 : continue
                res[i].append(self.findFace(img, detections))
        return res     

    def findFace(self, img, detections):
#         print("detections nums", len(detections))
        i = 0
        ymin = math.floor(detections[i, 0] * img.shape[0])
        xmin = math.floor(detections[i, 1] * img.shape[1])
        ymax = math.ceil(detections[i, 2] * img.shape[0])
        xmax = math.ceil(detections[i, 3] * img.shape[1])
#         print(detections[i, 0], detections[i, 1],detections[i, 2], detections[i, 3])
#         print(img.shape)
#         print(xmin, xmax, ymin, ymax)
        face = img[ymin:ymax, xmin:xmax+1]
#         print(face.shape)

        return face


class EfficientNetModel():
    def __init__(self,model_name, device, tfms=0):
        self.model = EfficientNet.from_name(model_name).to(device)
        self.model._fc = nn.Linear(self.model._fc.in_features, 1)
        self.device = device
        if tfms == 0:
            self.tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    
    def load_stat_dict(self, stat_file):
        stat_dict = torch.load(stat_file, map_location=self.device)
        self.model.load_state_dict(stat_dict)
    
    def transfrom(self, img):
        face = PIL.Image.fromarray(np.uint8(img))
        return self.tfms(face).unsqueeze(0)
    
    def train(self, trainX, batch_size, epoch, device, label, state_dict_file, optimizer=None,scheduler=None):
        trainX_arr = [img for k in range(len(trainX)) for img in trainX[k]]
#         print(len(trainX_arr))
#         print(trainX_arr)
        trainData = []
        for img in trainX_arr:
#             print(img.shape)
            try:
                img = PIL.Image.fromarray(img)
            except:
                continue
            img = self.tfms(img)
            trainData.append(img)
        trainX = DataLoader(trainData, batch_size=batch_size)
        self.model = self.model.to(device)
        self.model.train()
        for i in range(epoch):
            total_loss = 0
            for imgs in trainX:
#                 print(imgs.shape)
                imgs = imgs.to(device)
        #         print(imgs.shape)
                if label:
                    y = torch.ones((len(imgs),1)).to(device)
                else:
                    y = torch.zeros((len(imgs),1)).to(device)
                y_predict = self.model(imgs)
                loss = F.binary_cross_entropy(F.sigmoid(y_predict), y)
                optimizer.zero_grad() #判别器D的梯度归零
        #         loss.backward(retain_graph=True) #反向传播
                loss.backward()
                optimizer.step()
                total_loss += loss
            print("setp:", i, "loss:", total_loss)
            torch.save(self.model.state_dict(),state_dict_file+"epoch-{0}.pth".format(i))
        torch.save(self.model.state_dict(),state_dict_file+'.pth')
        
    def validate(self, validateX, state_dict_file):
        
        result = []
        num = len(validateX)
        for i in range(num):
            true_label = 0
            false_label = 0
            for img in validateX[i]:
                try:
                    img = PIL.Image.fromarray(np.uint8(img))
                except Exception as e:
#                     print(e)
                    continue
                img_p = self.tfms(img).unsqueeze(0)
                label = self.getLabelResult(img_p)
                if label == 0:
                    true_label += 1
                else:
                    false_label += 1
            if true_label > false_label:
                result.append(0)
            else:
                result.append(1)
        print("true: ", result.count(0), "false:", result.count(1), "total:", len(result))
            
        
        
        
    def getPResult(self, input_):
        input_ = input_.to(self.device)
        self.model = self.model.to(self.device)
        output = self.model(input_)
        output = torch.sigmoid(output)
#         print(output)
        return output
    
    def getLabelResult(self, input_):
        output = self.getPResult(input_)
        label = 0 if output < 0.5 else 1
        return label


# In[ ]:


import sys
video_process = VideoProcess()
videos_dir = "/kaggle/input/faceforensics/original_sequences/youtube/c23/videos"
deepfake_dir = "/kaggle/input/faceforensics/manipulated_sequences/Deepfakes/c23/videos"
faceswap_dir = "/kaggle/input/faceforensics/manipulated_sequences/FaceSwap/c23/videos"
face2face_dir = "/kaggle/input/faceforensics/manipulated_sequences/Face2Face/c23/videos"
neuralTextures_dir = "/kaggle/input/faceforensics/manipulated_sequences/NeuralTextures/c23/videos"


#     print(os.listdir(videos_dir))
print("origin getFrame start:")
imgs_origin = video_process.getFrame(videos_dir, file_num=500, frame_num=5, type=0)
print("origin GetFrame Over!")
print("origin Blaze start:")
res_origin = video_process.blazeFace(imgs_origin)
print("origin Blaze Over!")



# In[ ]:


print("deepfake getFrame start:")
imgs_deepfake = video_process.getFrame(deepfake_dir, file_num=500, frame_num=5, type=0)
print("deepfake GetFrame Over!")
print("deepfake Blaze start:")
res_deepfake = video_process.blazeFace(imgs_deepfake)
print("deepfake Blaze Over!")


# In[ ]:


model_name = 'efficientnet-b0'
model = EfficientNetModel(model_name, gpu)
stat_file = '/kaggle/input/deepfake-detection-model-20k/EfficientNetb0 t2 0.8616966359803837 0.3698434531609828.pth'
# stat_file = "/kaggle/input/deepfake-detection-model-20k/EfficientNetb1 t2 0.8410909403768391 0.36058002083572327.pth"
stat_file = "origin_videos.pth"
model.load_stat_dict(stat_file)
model.model.eval()
optimizer = torch.optim.Adam(model.model.parameters(), lr=0.0001, weight_decay=0.) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
model.validate(res_origin, state_dict_file=stat_file)
model.validate(res_deepfake, state_dict_file=stat_file)

model.train(res_origin[:400], batch_size=32, epoch=20, device=gpu, label=0, state_dict_file="origin_videos", optimizer=optimizer,scheduler=None)
model.train(res_deepfake[:400], batch_size=32, epoch=20, device=gpu, label=1, state_dict_file="origin_videos", optimizer=optimizer,scheduler=None)
# model.load_stat_dict("origin_videos.pth")

model.validate(res_origin[400:], state_dict_file="origin_videos.pth")
model.validate(res_deepfake[400:], state_dict_file="origin_videos.pth")


# In[ ]:


{"origin" : 264/236,
 "train"  : 89/100,
 "deepfake":  ,
}

