#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install visualize
#!pip install visdom


# In[ ]:


get_ipython().run_cell_magic('writefile', 'dataset.py', "#encoding:utf-8\n#\n#created by xiongzihua\n#\n'''\ntxt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标\n'''\nimport os\nimport sys\nimport os.path\n\nimport random\nimport numpy as np\n\nimport torch\nimport torch.utils.data as data\nimport torchvision.transforms as transforms\n\nimport cv2\nimport matplotlib.pyplot as plt\n\nclass yoloDataset(data.Dataset):\n    image_size = 448\n    def __init__(self,root,list_file,train,transform):\n        print('data init')\n        self.root=root\n        self.train = train\n        self.transform=transform\n        self.fnames = []\n        self.boxes = []\n        self.labels = []\n        self.mean = (123,117,104)#RGB\n\n#         if isinstance(list_file, list):\n#             # Cat multiple list files together.\n#             # This is especially useful for voc07/voc12 combination.\n#             open('./listfile.txt','w')\n#             tmp_file = './listfile.txt'\n#             os.system('cat %s > %s' % (' '.join(list_file), tmp_file))\n#             #os.system('type %s > %s' % (' '.join(list_file), tmp_file))\n#             list_file = tmp_file\n\n        with open(list_file) as f:\n            lines  = f.readlines()\n\n        for line in lines:\n            splited = line.strip().split()\n            self.fnames.append(splited[0])\n            num_boxes = (len(splited) - 1) // 5\n            box=[]\n            label=[]\n            for i in range(num_boxes):\n                x = float(splited[1+5*i])\n                y = float(splited[2+5*i])\n                x2 = float(splited[3+5*i])\n                y2 = float(splited[4+5*i])\n                c = splited[5+5*i]\n                box.append([x,y,x2,y2])\n                label.append(int(c)+1)\n            self.boxes.append(torch.Tensor(box))\n            self.labels.append(torch.LongTensor(label))\n        self.num_samples = len(self.boxes)\n\n    def __getitem__(self,idx):\n        fname = self.fnames[idx]\n        img = cv2.imread(os.path.join(self.root+fname))\n        boxes = self.boxes[idx].clone()\n        labels = self.labels[idx].clone()\n\n        if self.train:\n            #img = self.random_bright(img)\n            img, boxes = self.random_flip(img, boxes)\n            img,boxes = self.randomScale(img,boxes)\n            img = self.randomBlur(img)\n            img = self.RandomBrightness(img)\n            img = self.RandomHue(img)\n            img = self.RandomSaturation(img)\n            img,boxes,labels = self.randomShift(img,boxes,labels)\n            img,boxes,labels = self.randomCrop(img,boxes,labels)\n        # #debug\n        # box_show = boxes.numpy().reshape(-1)\n        # print(box_show)\n        # img_show = self.BGR2RGB(img)\n        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))\n        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)\n        # plt.figure()\n        \n        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)\n        # plt.imshow(img_show)\n        # plt.show()\n        # #debug\n        h,w,_ = img.shape\n        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)\n        img = self.BGR2RGB(img) #because pytorch pretrained model use RGB\n        img = self.subMean(img,self.mean) #减去均值\n        img = cv2.resize(img,(self.image_size,self.image_size))\n        target = self.encoder(boxes,labels)# 7x7x30\n        for t in self.transform:\n            img = t(img)\n\n        return img,target\n    def __len__(self):\n        return self.num_samples\n\n    def encoder(self,boxes,labels):\n        '''\n        boxes (tensor) [[x1,y1,x2,y2],[]]\n        labels (tensor) [...]\n        return 7x7x30\n        '''\n        grid_num = 14\n        target = torch.zeros((grid_num,grid_num,34))\n        cell_size = 1./grid_num\n        wh = boxes[:,2:]-boxes[:,:2]\n        cxcy = (boxes[:,2:]+boxes[:,:2])/2\n        for i in range(cxcy.size()[0]):\n            cxcy_sample = cxcy[i]\n            ij = (cxcy_sample/cell_size).ceil()-1 #\n            target[int(ij[1]),int(ij[0]),4] = 1\n            target[int(ij[1]),int(ij[0]),9] = 1\n            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1\n            xy = ij*cell_size #匹配到的网格的左上角相对坐标\n            delta_xy = (cxcy_sample -xy)/cell_size\n            target[int(ij[1]),int(ij[0]),2:4] = wh[i]\n            target[int(ij[1]),int(ij[0]),:2] = delta_xy\n            target[int(ij[1]),int(ij[0]),7:9] = wh[i]\n            target[int(ij[1]),int(ij[0]),5:7] = delta_xy\n        return target\n    def BGR2RGB(self,img):\n        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)\n    def BGR2HSV(self,img):\n        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)\n    def HSV2BGR(self,img):\n        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)\n    \n    def RandomBrightness(self,bgr):\n        if random.random() < 0.5:\n            hsv = self.BGR2HSV(bgr)\n            h,s,v = cv2.split(hsv)\n            adjust = random.choice([0.5,1.5])\n            v = v*adjust\n            v = np.clip(v, 0, 255).astype(hsv.dtype)\n            hsv = cv2.merge((h,s,v))\n            bgr = self.HSV2BGR(hsv)\n        return bgr\n    def RandomSaturation(self,bgr):\n        if random.random() < 0.5:\n            hsv = self.BGR2HSV(bgr)\n            h,s,v = cv2.split(hsv)\n            adjust = random.choice([0.5,1.5])\n            s = s*adjust\n            s = np.clip(s, 0, 255).astype(hsv.dtype)\n            hsv = cv2.merge((h,s,v))\n            bgr = self.HSV2BGR(hsv)\n        return bgr\n    def RandomHue(self,bgr):\n        if random.random() < 0.5:\n            hsv = self.BGR2HSV(bgr)\n            h,s,v = cv2.split(hsv)\n            adjust = random.choice([0.5,1.5])\n            h = h*adjust\n            h = np.clip(h, 0, 255).astype(hsv.dtype)\n            hsv = cv2.merge((h,s,v))\n            bgr = self.HSV2BGR(hsv)\n        return bgr\n\n    def randomBlur(self,bgr):\n        if random.random()<0.5:\n            bgr = cv2.blur(bgr,(5,5))\n        return bgr\n\n    def randomShift(self,bgr,boxes,labels):\n        #平移变换\n        center = (boxes[:,2:]+boxes[:,:2])/2\n        if random.random() <0.5:\n            height,width,c = bgr.shape\n            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)\n            after_shfit_image[:,:,:] = (104,117,123) #bgr\n            shift_x = random.uniform(-width*0.2,width*0.2)\n            shift_y = random.uniform(-height*0.2,height*0.2)\n            #print(bgr.shape,shift_x,shift_y)\n            #原图像的平移\n            if shift_x>=0 and shift_y>=0:\n                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]\n            elif shift_x>=0 and shift_y<0:\n                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]\n            elif shift_x <0 and shift_y >=0:\n                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]\n            elif shift_x<0 and shift_y<0:\n                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]\n\n            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)\n            center = center + shift_xy\n            mask1 = (center[:,0] >0) & (center[:,0] < width)\n            mask2 = (center[:,1] >0) & (center[:,1] < height)\n            mask = (mask1 & mask2).view(-1,1)\n            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)\n            if len(boxes_in) == 0:\n                return bgr,boxes,labels\n            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)\n            boxes_in = boxes_in+box_shift\n            labels_in = labels[mask.view(-1)]\n            return after_shfit_image,boxes_in,labels_in\n        return bgr,boxes,labels\n\n    def randomScale(self,bgr,boxes):\n        #固定住高度，以0.8-1.2伸缩宽度，做图像形变\n        if random.random() < 0.5:\n            scale = random.uniform(0.8,1.2)\n            height,width,c = bgr.shape\n            bgr = cv2.resize(bgr,(int(width*scale),height))\n            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)\n            boxes = boxes * scale_tensor\n            return bgr,boxes\n        return bgr,boxes\n\n    def randomCrop(self,bgr,boxes,labels):\n        if random.random() < 0.5:\n            center = (boxes[:,2:]+boxes[:,:2])/2\n            height,width,c = bgr.shape\n            h = random.uniform(0.6*height,height)\n            w = random.uniform(0.6*width,width)\n            x = random.uniform(0,width-w)\n            y = random.uniform(0,height-h)\n            x,y,h,w = int(x),int(y),int(h),int(w)\n\n            center = center - torch.FloatTensor([[x,y]]).expand_as(center)\n            mask1 = (center[:,0]>0) & (center[:,0]<w)\n            mask2 = (center[:,1]>0) & (center[:,1]<h)\n            mask = (mask1 & mask2).view(-1,1)\n\n            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)\n            if(len(boxes_in)==0):\n                return bgr,boxes,labels\n            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)\n\n            boxes_in = boxes_in - box_shift\n            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)\n            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)\n            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)\n            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)\n\n            labels_in = labels[mask.view(-1)]\n            img_croped = bgr[y:y+h,x:x+w,:]\n            return img_croped,boxes_in,labels_in\n        return bgr,boxes,labels\n\n\n\n\n    def subMean(self,bgr,mean):\n        mean = np.array(mean, dtype=np.float32)\n        bgr = bgr - mean\n        return bgr\n\n    def random_flip(self, im, boxes):\n        if random.random() < 0.5:\n            #im_lr = np.fliplr(im).copy()\n            im_lr = np.flip(im).copy()\n            h,w,_ = im.shape\n            xmin = w - boxes[:,2]\n            xmax = w - boxes[:,0]\n            boxes[:,0] = xmin\n            boxes[:,2] = xmax\n            return im_lr, boxes\n        return im, boxes\n    def random_bright(self, im, delta=16):\n        alpha = random.random()\n        if alpha > 0.3:\n            im = im * alpha + random.randrange(-delta,delta)\n            im = im.clip(min=0,max=255).astype(np.uint8)\n        return im\n\ndef main():\n    from torch.utils.data import DataLoader\n    import torchvision.transforms as transforms\n    file_root = '../input/gesture/train/images/'\n    train_dataset = yoloDataset(root=file_root,list_file='../input/vocgesture/vocgesturetest.txt',train=True,transform = [transforms.ToTensor()] )\n    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)\n    train_iter = iter(train_loader)\n    for i in range(100):\n        img,target = next(train_iter)\n        print(img,target)\n\n\nif __name__ == '__main__':\n    main()\n\n")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'net.py', '#encoding:utf-8\nimport torch.nn as nn\nimport torch.utils.model_zoo as model_zoo\nimport math\nimport torch.nn.functional as F\n\n\n__all__ = [\n    \'VGG\', \'vgg11\', \'vgg11_bn\', \'vgg13\', \'vgg13_bn\', \'vgg16\', \'vgg16_bn\',\n    \'vgg19_bn\', \'vgg19\',\n]\n\n\nmodel_urls = {\n    \'vgg11\': \'https://download.pytorch.org/models/vgg11-bbd30ac9.pth\',\n    \'vgg13\': \'https://download.pytorch.org/models/vgg13-c768596a.pth\',\n    \'vgg16\': \'https://download.pytorch.org/models/vgg16-397923af.pth\',\n    \'vgg19\': \'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth\',\n    \'vgg11_bn\': \'https://download.pytorch.org/models/vgg11_bn-6002323d.pth\',\n    \'vgg13_bn\': \'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth\',\n    \'vgg16_bn\': \'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth\',\n    \'vgg19_bn\': \'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth\',\n}\n\n\nclass VGG(nn.Module):\n\n    def __init__(self, features, num_classes=1000, image_size=448):\n        super(VGG, self).__init__()\n        self.features = features\n        self.image_size = image_size\n        # self.classifier = nn.Sequential(\n        #     nn.Linear(512 * 7 * 7, 4096),\n        #     nn.ReLU(True),\n        #     nn.Dropout(),\n        #     nn.Linear(4096, 4096),\n        #     nn.ReLU(True),\n        #     nn.Dropout(),\n        #     nn.Linear(4096, num_classes),\n        # )\n        # if self.image_size == 448:\n        #     self.extra_conv1 = conv_bn_relu(512,512)\n        #     self.extra_conv2 = conv_bn_relu(512,512)\n        #     self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)\n        self.classifier = nn.Sequential(\n            nn.Linear(512 * 7 * 7, 4096),\n            nn.ReLU(True),\n            nn.Dropout(),\n            nn.Linear(4096, 1470),\n        )\n        self._initialize_weights()\n\n    def forward(self, x):\n        x = self.features(x)\n        # if self.image_size == 448:\n        #     x = self.extra_conv1(x)\n        #     x = self.extra_conv2(x)\n        #     x = self.downsample(x)\n        x = x.view(x.size(0), -1)\n        x = self.classifier(x)\n        x = F.sigmoid(x) #归一化到0-1\n        x = x.view(-1,7,7,30)\n        return x\n\n    def _initialize_weights(self):\n        for m in self.modules():\n            if isinstance(m, nn.Conv2d):\n                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels\n                m.weight.data.normal_(0, math.sqrt(2. / n))\n                if m.bias is not None:\n                    m.bias.data.zero_()\n            elif isinstance(m, nn.BatchNorm2d):\n                m.weight.data.fill_(1)\n                m.bias.data.zero_()\n            elif isinstance(m, nn.Linear):\n                m.weight.data.normal_(0, 0.01)\n                m.bias.data.zero_()\n\n\ndef make_layers(cfg, batch_norm=False):\n    layers = []\n    in_channels = 3\n    s = 1\n    first_flag=True\n    for v in cfg:\n        s=1\n        if (v==64 and first_flag):\n            s=2\n            first_flag=False\n        if v == \'M\':\n            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]\n        else:\n            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)\n            if batch_norm:\n                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]\n            else:\n                layers += [conv2d, nn.ReLU(inplace=True)]\n            in_channels = v\n    return nn.Sequential(*layers)\n\ndef conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):\n    return nn.Sequential(\n        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),\n        nn.BatchNorm2d(out_channels),\n        nn.ReLU(True)\n    )\n\n\ncfg = {\n    \'A\': [64, \'M\', 128, \'M\', 256, 256, \'M\', 512, 512, \'M\', 512, 512, \'M\'],\n    \'B\': [64, 64, \'M\', 128, 128, \'M\', 256, 256, \'M\', 512, 512, \'M\', 512, 512, \'M\'],\n    \'D\': [64, 64, \'M\', 128, 128, \'M\', 256, 256, 256, \'M\', 512, 512, 512, \'M\', 512, 512, 512, \'M\'],\n    \'E\': [64, 64, \'M\', 128, 128, \'M\', 256, 256, 256, 256, \'M\', 512, 512, 512, 512, \'M\', 512, 512, 512, 512, \'M\'],\n}\n\n\ndef vgg11(pretrained=False, **kwargs):\n    """VGG 11-layer model (configuration "A")\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = VGG(make_layers(cfg[\'A\']), **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'vgg11\']))\n    return model\n\n\ndef vgg11_bn(pretrained=False, **kwargs):\n    """VGG 11-layer model (configuration "A") with batch normalization\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = VGG(make_layers(cfg[\'A\'], batch_norm=True), **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'vgg11_bn\']))\n    return model\n\n\ndef vgg13(pretrained=False, **kwargs):\n    """VGG 13-layer model (configuration "B")\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = VGG(make_layers(cfg[\'B\']), **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'vgg13\']))\n    return model\n\n\ndef vgg13_bn(pretrained=False, **kwargs):\n    """VGG 13-layer model (configuration "B") with batch normalization\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = VGG(make_layers(cfg[\'B\'], batch_norm=True), **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'vgg13_bn\']))\n    return model\n\n\ndef vgg16(pretrained=False, **kwargs):\n    """VGG 16-layer model (configuration "D")\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = VGG(make_layers(cfg[\'D\']), **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'vgg16\']))\n    return model\n\n\ndef vgg16_bn(pretrained=False, **kwargs):\n    """VGG 16-layer model (configuration "D") with batch normalization\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = VGG(make_layers(cfg[\'D\'], batch_norm=True), **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'vgg16_bn\']))\n    return model\n\n\ndef vgg19(pretrained=False, **kwargs):\n    """VGG 19-layer model (configuration "E")\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = VGG(make_layers(cfg[\'E\']), **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'vgg19\']))\n    return model\n\n\ndef vgg19_bn(pretrained=False, **kwargs):\n    """VGG 19-layer model (configuration \'E\') with batch normalization\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = VGG(make_layers(cfg[\'E\'], batch_norm=True), **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'vgg19_bn\']))\n    return model\n\ndef test():\n    import torch\n    from torch.autograd import Variable\n    model = vgg16()\n    model.classifier = nn.Sequential(\n            nn.Linear(512 * 7 * 7, 4096),\n            nn.ReLU(True),\n            nn.Dropout(),\n            nn.Linear(4096, 4096),\n            nn.ReLU(True),\n            nn.Dropout(),\n            nn.Linear(4096, 1470),\n        )\n    print(model.classifier[6]) \n    #print(model)\n    img = torch.rand(2,3,224,224)\n    img = Variable(img)\n    output = model(img)\n    print(output.size())\n\nif __name__ == \'__main__\':\n    test()')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'resnet_yolo.py', 'import torch.nn as nn\nimport math\nimport torch.utils.model_zoo as model_zoo\nimport torch.nn.functional as F\n\n\n__all__ = [\'ResNet\', \'resnet18\', \'resnet34\', \'resnet50\', \'resnet101\',\n           \'resnet152\']\n\n\nmodel_urls = {\n    \'resnet18\': \'https://download.pytorch.org/models/resnet18-5c106cde.pth\',\n    \'resnet34\': \'https://download.pytorch.org/models/resnet34-333f7ec4.pth\',\n    \'resnet50\': \'https://download.pytorch.org/models/resnet50-19c8e357.pth\',\n    \'resnet101\': \'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth\',\n    \'resnet152\': \'https://download.pytorch.org/models/resnet152-b121ed2d.pth\',\n}\n\n\ndef conv3x3(in_planes, out_planes, stride=1):\n    "3x3 convolution with padding"\n    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,\n                     padding=1, bias=False)\n\n\nclass BasicBlock(nn.Module):\n    expansion = 1\n\n    def __init__(self, inplanes, planes, stride=1, downsample=None):\n        super(BasicBlock, self).__init__()\n        self.conv1 = conv3x3(inplanes, planes, stride)\n        self.bn1 = nn.BatchNorm2d(planes)\n        self.relu = nn.ReLU(inplace=True)\n        self.conv2 = conv3x3(planes, planes)\n        self.bn2 = nn.BatchNorm2d(planes)\n        self.downsample = downsample\n        self.stride = stride\n\n    def forward(self, x):\n        residual = x\n\n        out = self.conv1(x)\n        out = self.bn1(out)\n        out = self.relu(out)\n\n        out = self.conv2(out)\n        out = self.bn2(out)\n\n        if self.downsample is not None:\n            residual = self.downsample(x)\n\n        out += residual\n        out = self.relu(out)\n\n        return out\n\n\nclass Bottleneck(nn.Module):\n    expansion = 4\n\n    def __init__(self, inplanes, planes, stride=1, downsample=None):\n        super(Bottleneck, self).__init__()\n        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)\n        self.bn1 = nn.BatchNorm2d(planes)\n        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,\n                               padding=1, bias=False)\n        self.bn2 = nn.BatchNorm2d(planes)\n        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)\n        self.bn3 = nn.BatchNorm2d(planes * 4)\n        self.relu = nn.ReLU(inplace=True)\n        self.downsample = downsample\n        self.stride = stride\n\n    def forward(self, x):\n        residual = x\n\n        out = self.conv1(x)\n        out = self.bn1(out)\n        out = self.relu(out)\n\n        out = self.conv2(out)\n        out = self.bn2(out)\n        out = self.relu(out)\n\n        out = self.conv3(out)\n        out = self.bn3(out)\n\n        if self.downsample is not None:\n            residual = self.downsample(x)\n\n        out += residual\n        out = self.relu(out)\n\n        return out\n\nclass detnet_bottleneck(nn.Module):\n    # no expansion\n    # dilation = 2\n    # type B use 1x1 conv\n    expansion = 1\n\n    def __init__(self, in_planes, planes, stride=1, block_type=\'A\'):\n        super(detnet_bottleneck, self).__init__()\n        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)\n        self.bn1 = nn.BatchNorm2d(planes)\n        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)\n        self.bn2 = nn.BatchNorm2d(planes)\n        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)\n        self.bn3 = nn.BatchNorm2d(self.expansion*planes)\n\n        self.downsample = nn.Sequential()\n        if stride != 1 or in_planes != self.expansion*planes or block_type==\'B\':\n            self.downsample = nn.Sequential(\n                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),\n                nn.BatchNorm2d(self.expansion*planes)\n            )\n\n    def forward(self, x):\n        out = F.relu(self.bn1(self.conv1(x)))\n        out = F.relu(self.bn2(self.conv2(out)))\n        out = self.bn3(self.conv3(out))\n        out += self.downsample(x)\n        out = F.relu(out)\n        return out\n\nclass ResNet(nn.Module):\n\n    def __init__(self, block, layers, num_classes=1470):\n        self.inplanes = 64\n        super(ResNet, self).__init__()\n        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,\n                               bias=False)\n        self.bn1 = nn.BatchNorm2d(64)\n        self.relu = nn.ReLU(inplace=True)\n        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)\n        self.layer1 = self._make_layer(block, 64, layers[0])\n        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)\n        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)\n        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)\n        # self.layer5 = self._make_layer(block, 512, layers[3], stride=2)\n        self.layer5 = self._make_detnet_layer(in_channels=2048)\n        # self.avgpool = nn.AvgPool2d(14) #fit 448 input size\n        # self.fc = nn.Linear(512 * block.expansion, num_classes)\n        # self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)\n        # self.bn_end = nn.BatchNorm2d(30)\n        self.conv_end = nn.Conv2d(256, 34, kernel_size=3, stride=1, padding=1, bias=False)\n        self.bn_end = nn.BatchNorm2d(34)\n        for m in self.modules():\n            if isinstance(m, nn.Conv2d):\n                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels\n                m.weight.data.normal_(0, math.sqrt(2. / n))\n            elif isinstance(m, nn.BatchNorm2d):\n                m.weight.data.fill_(1)\n                m.bias.data.zero_()\n\n    def _make_layer(self, block, planes, blocks, stride=1):\n        downsample = None\n        if stride != 1 or self.inplanes != planes * block.expansion:\n            downsample = nn.Sequential(\n                nn.Conv2d(self.inplanes, planes * block.expansion,\n                          kernel_size=1, stride=stride, bias=False),\n                nn.BatchNorm2d(planes * block.expansion),\n            )\n\n        layers = []\n        layers.append(block(self.inplanes, planes, stride, downsample))\n        self.inplanes = planes * block.expansion\n        for i in range(1, blocks):\n            layers.append(block(self.inplanes, planes))\n\n        return nn.Sequential(*layers)\n    \n    def _make_detnet_layer(self,in_channels):\n        layers = []\n        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type=\'B\'))\n        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type=\'A\'))\n        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type=\'A\'))\n        return nn.Sequential(*layers)\n\n    def forward(self, x):\n        x = self.conv1(x)\n        x = self.bn1(x)\n        x = self.relu(x)\n        x = self.maxpool(x)\n\n        x = self.layer1(x)\n        x = self.layer2(x)\n        x = self.layer3(x)\n        x = self.layer4(x)\n        x = self.layer5(x)\n        # x = self.avgpool(x)\n        # x = x.view(x.size(0), -1)\n        # x = self.fc(x)\n        x = self.conv_end(x)\n        x = self.bn_end(x)\n        x = F.sigmoid(x) #归一化到0-1\n        # x = x.view(-1,7,7,30)\n        x = x.permute(0,2,3,1) #(-1,7,7,30)\n\n        return x\n\n\ndef resnet18(pretrained=False, **kwargs):\n    """Constructs a ResNet-18 model.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'resnet18\']))\n    return model\n\n\ndef resnet34(pretrained=False, **kwargs):\n    """Constructs a ResNet-34 model.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'resnet34\']))\n    return model\n\n\ndef resnet50(pretrained=False, **kwargs):\n    """Constructs a ResNet-50 model.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'resnet50\']))\n    return model\n\n\ndef resnet101(pretrained=False, **kwargs):\n    """Constructs a ResNet-101 model.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'resnet101\']))\n    return model\n\n\ndef resnet152(pretrained=False, **kwargs):\n    """Constructs a ResNet-152 model.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n    """\n    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)\n    if pretrained:\n        model.load_state_dict(model_zoo.load_url(model_urls[\'resnet152\']))\n    return model')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'yoloLoss.py', "#encoding:utf-8\n#\n#created by xiongzihua 2017.12.26\n#\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.autograd import Variable\n\nclass yoloLoss(nn.Module):\n    def __init__(self,S,B,l_coord,l_noobj):\n        super(yoloLoss,self).__init__()\n        self.S = S\n        self.B = B\n        self.l_coord = l_coord\n        self.l_noobj = l_noobj\n\n    def compute_iou(self, box1, box2):\n        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].\n        Args:\n          box1: (tensor) bounding boxes, sized [N,4].\n          box2: (tensor) bounding boxes, sized [M,4].\n        Return:\n          (tensor) iou, sized [N,M].\n        '''\n        N = box1.size(0)\n        M = box2.size(0)\n\n        lt = torch.max(\n            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]\n            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]\n        )\n\n        rb = torch.min(\n            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]\n            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]\n        )\n\n        wh = rb - lt  # [N,M,2]\n        wh[wh<0] = 0  # clip at 0\n        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]\n\n        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]\n        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]\n        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]\n        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]\n\n        iou = inter / (area1 + area2 - inter)\n        return iou\n    def forward(self,pred_tensor,target_tensor):\n        '''\n        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]\n        target_tensor: (tensor) size(batchsize,S,S,30)\n        '''\n        N = pred_tensor.size()[0]\n        coo_mask = target_tensor[:,:,:,4] > 0\n        noo_mask = target_tensor[:,:,:,4] == 0\n        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)\n        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)\n\n        coo_pred = pred_tensor[coo_mask].view(-1,30)\n        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]\n        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]\n        \n        coo_target = target_tensor[coo_mask].view(-1,30)\n        box_target = coo_target[:,:10].contiguous().view(-1,5)\n        class_target = coo_target[:,10:]\n\n        # compute not contain obj loss\n        noo_pred = pred_tensor[noo_mask].view(-1,30)\n        noo_target = target_tensor[noo_mask].view(-1,30)\n        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()).bool()\n        #noo_pred_mask = torch.ByteTensor(noo_pred.size())\n        noo_pred_mask.zero_()\n        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1\n        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]\n        noo_target_c = noo_target[noo_pred_mask]\n        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)\n\n        #compute contain obj loss\n        coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()\n        #coo_response_mask = torch.ByteTensor(box_target.size())\n        coo_response_mask.zero_()\n        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()\n        #coo_not_response_mask = torch.ByteTensor(box_target.size())\n        coo_not_response_mask.zero_()\n        box_target_iou = torch.zeros(box_target.size()).cuda()\n        #box_target_iou = torch.zeros(box_target.size())\n        for i in range(0,box_target.size()[0],2): #choose the best iou box\n            box1 = box_pred[i:i+2]\n            box1_xyxy = Variable(torch.FloatTensor(box1.size()))\n            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]\n            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]\n            box2 = box_target[i].view(-1,5)\n            box2_xyxy = Variable(torch.FloatTensor(box2.size()))\n            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]\n            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]\n            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]\n            max_iou,max_index = iou.max(0)\n            max_index = max_index.data.cuda()\n            #max_index = max_index.data\n            \n            coo_response_mask[i+max_index]=1\n            coo_not_response_mask[i+1-max_index]=1\n\n            #####\n            # we want the confidence score to equal the\n            # intersection over union (IOU) between the predicted box\n            # and the ground truth\n            #####\n            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()\n            #box_target_iou[i+max_index,torch.LongTensor([4])] = (max_iou).data\n        box_target_iou = Variable(box_target_iou).cuda()\n        #box_target_iou = Variable(box_target_iou)\n        #1.response loss\n        box_pred_response = box_pred[coo_response_mask].view(-1,5)\n        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)\n        box_target_response = box_target[coo_response_mask].view(-1,5)\n        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)\n        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)\n        #2.not response loss\n        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)\n        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)\n        box_target_not_response[:,4]= 0\n        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)\n        \n        #I believe this bug is simply a typo\n        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)\n\n        #3.class loss\n        class_loss = F.mse_loss(class_pred,class_target,size_average=False)\n\n        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N\n\n\n\n")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'visualize1.py', "import visdom \nimport numpy as np\n\nclass Visualizer():\n    def __init__(self, env='main', **kwargs):\n        '''\n        **kwargs, dict option\n        '''\n        self.vis = visdom.Visdom(env=env)\n        self.index = {}  # x, dict\n        self.log_text = ''\n        self.env = env\n    \n    def plot_train_val(self, loss_train=None, loss_val=None):\n        '''\n        plot val loss and train loss in one figure\n        '''\n        x = self.index.get('train_val', 0)\n\n        if x == 0:\n            loss = loss_train if loss_train else loss_val\n            win_y = np.column_stack((loss, loss))\n            win_x = np.column_stack((x, x))\n            self.win = self.vis.line(Y=win_y, X=win_x, \n                                env=self.env)\n                                # opts=dict(\n                                #     title='train_test_loss',\n                                # ))\n            self.index['train_val'] = x + 1\n            return \n\n        if loss_train != None:\n            self.vis.line(Y=np.array([loss_train]), X=np.array([x]),\n                        win=self.win,\n                        name='1',\n                        update='append',\n                        env=self.env)\n            self.index['train_val'] = x + 5\n        else:\n            self.vis.line(Y=np.array([loss_val]), X=np.array([x]),\n                        win=self.win,\n                        name='2',\n                        update='append',\n                        env=self.env)\n\n    def plot_many(self, d):\n        '''\n        d: dict {name, value}\n        '''\n        for k, v in d.iteritems():\n            self.plot(k, v)\n\n    def plot(self, name, y, **kwargs):\n        '''\n        plot('loss', 1.00)\n        '''\n        x = self.index.get(name, 0) # if none, return 0\n        self.vis.line(Y=np.array([y]), X=np.array([x]),\n                    win=name,\n                    opts=dict(title=name),\n                    update=None if x== 0 else 'append',\n                    **kwargs)\n        self.index[name] = x + 1\n    \n    def log(self, info, win='log_text'):\n        '''\n        show text in box not write into txt?\n        '''\n        pass")


# In[ ]:


#train.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset

#from visualize1 import Visualizer
import numpy as np

import warnings
warnings.filterwarnings('ignore')

use_gpu = torch.cuda.is_available()

file_root = '../input/gesture/train/images/'
learning_rate = 0.001
num_epochs = 1
batch_size = 24
use_resnet = True
if use_resnet:
    net = resnet50()
else:
    net = vgg16_bn()
# net.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             #nn.Linear(4096, 4096),
#             #nn.ReLU(True),
#             #nn.Dropout(),
#             nn.Linear(4096, 1470),
#         )
#net = resnet18(pretrained=True)
#net.fc = nn.Linear(512,1470)
# initial Linear
# for m in net.modules():
#     if isinstance(m, nn.Linear):
#         m.weight.data.normal_(0, 0.01)
#         m.bias.data.zero_()
print(net)
#net.load_state_dict(torch.load('yolo.pth'))
print('load pre-trined model')
if use_resnet:
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in dd.keys() and not k.startswith('fc'):
            print('yes')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
else:
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in dd.keys() and k.startswith('features'):
            print('yes')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
if False:
    net.load_state_dict(torch.load('best.pth'))
# print('cuda', torch.cuda.current_device(), torch.cuda.device_count())


criterion = yoloLoss(7,2,5,0.5)
if use_gpu:
    net.cuda()

net.train()
# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

# train_dataset = yoloDataset(root=file_root,list_file=['voc12_trainval.txt','voc07_trainval.txt'],train=True,transform = [transforms.ToTensor()] )
train_dataset = yoloDataset(root=file_root,list_file='../input/vocgesture/vocgesturetest.txt',train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
# test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
# test_dataset = yoloDataset(root=file_root,list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
# test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('./log.txt', 'w')

num_iter = 0
#vis = Visualizer(env='xiong')
best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
    # if epoch == 1:
    #     learning_rate = 0.0005
    # if epoch == 2:
    #     learning_rate = 0.00075
    # if epoch == 3:
    #     learning_rate = 0.001
    if epoch == 30:
        learning_rate=0.0001
    if epoch == 40:
        learning_rate=0.00001
    # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        # total_loss += loss.data[0]
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1
           # vis.plot_train_val(loss_train=total_loss/(i+1))

    #validation
    validation_loss = 0.0
    net.eval()
#     for i,(images,target) in enumerate(test_loader):
#         images = Variable(images,volatile=True)
#         target = Variable(target,volatile=True)
#         if use_gpu:
#             images,target = images.cuda(),target.cuda()
        
#         pred = net(images)
#         loss = criterion(pred,target)
#         validation_loss += loss.item()
#     validation_loss /= len(test_loader)
   # vis.plot_train_val(loss_val=validation_loss)
    
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(),'best.pth')
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
    logfile.flush()      
    torch.save(net.state_dict(),'yolo.pth')
    

#train.py

