#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


trainData = pd.read_csv('/kaggle/input/ml2021-2022-2-cnn/train.csv')
testData = pd.read_csv('/kaggle/input/ml2021-2022-2-cnn/test.csv')

train_x = np.array(trainData)[:,1:]
train_y = np.array(trainData)[:,0]
test_x = np.array(testData)
print('train_x:',train_x.shape)
print('train_y:',train_y.shape)
print('test_x:',test_x.shape)


# # 处理数据

# ## 2.1转为二维图像

# In[ ]:


H , W = int(train_x.shape[1]**0.5) , int(train_x.shape[1]**0.5)
train_imgs = train_x.reshape(-1,H,W)
test_imgs = test_x.reshape(-1,H,W)

print('train_x:',train_imgs.shape)
print('test_x:',test_imgs.shape)


# ## 2.2可视化图像

# In[ ]:


#展示数字
from matplotlib import pyplot as plt 

def showImage(image):
    plt.gray()
    plt.imshow(image)
    plt.show()
    
show_num = 5
for i in range(show_num):
    showImage(train_imgs[i])
    print('数字是:',train_y[i])


# ## 2.3归一化

# In[ ]:


def normalization(x):
    eps = 1e-5
    if x.ndim > 2:
        mean = np.mean(x, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]
        var = np.var(x, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]
        x = (x - mean) / np.sqrt(var + eps)
    else:
        mean = np.mean(x, axis=1)[:, np.newaxis]
        var = np.var(x, axis=1)[:, np.newaxis] + eps
        x = (x - mean) / np.sqrt(var)

    return x


# # 3 多个模块的实现

# ## 3.1参数结构体

# In[ ]:


class parameter():
    def __init__(self, w):
        self.data = w     # 权重
        self.grad = None  # 传到下一层的梯度


# ## 3.2卷积层

# ### 3.2.1卷积

# In[ ]:


class conv():
    def __init__(self, filter_shape, stride=1, padding='SAME', bias=True, requires_grad=True):
        """
        :param filter_shape:元组（O, C, K, K）
        :param stride: 步长
        :param padding: 填充方式:{"SAME", "VALID"}
        :param bias:是否有偏置
        :param requires_grad:是否计算梯度
        """
        self.weight = parameter(np.random.randn(*filter_shape) * (2 / reduce(lambda x,y: x*y, filter_shape[1:])**0.5))  # #kaiming初始化
        self.stride = stride
        self.padding = padding
        self.requires_grad = requires_grad
        self.output_channel = filter_shape[0]   # 输出通道数
        self.input_channel = filter_shape[1]    # 输入通道数
        self.filter_size = filter_shape[2]  # 卷积核大小
        if bias:
            self.bias = parameter(np.random.randn(self.output_channel))
        else:
            self.bias = None


    def forward(self, input):
        """
        :param input:feature map维度：[N,C,H,W]
        :return:卷积结果result：[N,O,output_H, output_W]
        """
        # 不做边缘填充
        if self.padding == "VALID":
            self.x = input
        # 填充
        elif self.padding == "SAME":
            p = self.filter_size // 2
            self.x = np.pad(input, ((0,0), (0,0), (p,p), (p,p)), "constant")

        """
        输入的宽高不能恰好的被卷积核的大小和选定的步长所整除时，有几个具体解决的策略：
        1、直接抛出异常；
        2、直接抛弃掉多余部分；
        3、边缘填0，使其满足要求等等
        """
        x_fit = (self.x.shape[2]-self.filter_size)%self.stride
        y_fit = (self.x.shape[3]-self.filter_size)%self.stride

        # if x_fit!=0 or y_fit!=0:
        #     print("input tensor width\height can\'t fit stride")
        #     return

        if (self.stride>1):
            if x_fit != 0:
                self.x = self.x[:,:,0:self.x.shape[2]-x_fit,:]
            if y_fit != 0:
                self.x = self.x[:,:,:,0:self.x.shape[3]-y_fit]


        # 卷积运算实现
        N, C, H, W = self.x.shape
        output_H, output_W = (H-self.filter_size)//self.stride+1, (W-self.filter_size)//self.stride+1
        result = np.zeros((N, self.output_channel, output_H, output_W))

        for n in range(N):
            for o in range(self.output_channel):
                for i in range(0, H-self.filter_size+1, self.stride):
                    for j in range(0, W-self.filter_size+1, self.stride):
                        result[n,o,i,j] = np.sum(self.x[n, :, i:i+self.filter_size, j:j+self.filter_size]
                                                 * self.weight.data[o,:,:,:])\
                                          + (self.bias.data[o] if self.bias else 0)

        return result


    def backward(self,eta, lr):
        """
        :param eta:上一层返回的梯度[N,O,output_H, output_W]
        :return:本层的梯度result
        说明：对于某一层conv层进行求导时分为两个部分：1、本层梯度反向传播到上一层；2、本层内求导，分别对 W,b
        """

        # 在实现卷积的反向传播中，有两点需要注意：1、当步长大于1时，上一层返回的梯度要行、列之间插0；
        # 2、对于“VALID”填充方式，要在梯度周围添加self.filter_size-1圈零；对于“SAME”填充方式，要在梯度周围添加self.filter_size//2 圈零
        if self.stride>1:
            N, O, output_H, output_W = eta.shape[:]
            inserted_H, inserted_W = output_H + (self.stride-1)*(output_H-1), output_W + (self.stride-1)*(output_W-1)
            insert_eta = np.zeros((N, O, inserted_H, inserted_W))
            insert_eta[:,:,::self.stride, ::self.stride] = eta[:]
            eta = insert_eta

        # 本层内求导，分别对 W,b
        N, _, H, W = eta.shape
        self.b_grad = eta.sum(axis=(0,2,3))
        self.W_grad = np.zeros(self.weight.data.shape)      # 形状[O, C, K, K]
        for i in range(self.filter_size):
            for j in range(self.filter_size):
                self.W_grad[:,:,i,j] = np.tensordot(eta, self.x[:,:,i:i+H,j:j+W], ([0,2,3], [0,2,3]))
        # 权重更新
        self.weight.data -= lr * self.W_grad / N
        self.bias.data -= lr * self.b_grad / N

        # 第二步边缘填充
        if self.padding == "VALID":
            p = self.filter_size - 1
            pad_eta = np.lib.pad(eta, ((0,0),(0,0),(p,p),(p,p)), "constant", constant_values=0)
            eta = pad_eta
        if self.padding == "SAME":
            p = self.filter_size // 2
            pad_eta = np.lib.pad(eta, ((0,0),(0,0),(p,p),(p,p)), "constant", constant_values=0)
            eta = pad_eta


        # 本层梯度反向传播到上一层
        weight_flip = np.flip(self.weight.data, (2,3))  # 卷积核旋转180度
        weight_flip_swap = np.swapaxes(weight_flip, 0, 1)  # 交换输入、输出通道的顺序[C,O,H,W]
        N, O, H, W = eta.shape
        output_H, output_W = (H-self.filter_size)//self.stride+1, (W-self.filter_size)//self.stride+1
        self.weight.grad = np.zeros((N, weight_flip_swap.shape[0], output_H, output_W))

        for n in range(N):
            for c in range(weight_flip_swap.shape[0]):
                for i in range(0, H-self.filter_size+1, self.stride):
                    for j in range(0, W-self.filter_size+1, self.stride):
                        self.weight.grad[n,c,i,j] = np.sum(eta[n, :, i:i+self.filter_size, j:j+self.filter_size]
                                                 * weight_flip_swap[c,:,:,:])

        return self.weight.grad


# ### 3.2.2快速卷积

# In[ ]:


class conv_fast():
    def __init__(self, filter_shape, stride=1, padding='SAME', bias=True, requires_grad=True):
        """
        :param filter_shape:元组（O, C, K, K）
        :param stride: 步长
        :param padding: 填充方式:{"SAME", "VALID"}
        :param bias:是否有偏置
        :param requires_grad:是否计算梯度
        """
        self.weight = parameter(np.random.randn(*filter_shape) * (2/reduce(lambda x,y:x*y, filter_shape[1:]))**0.5)  #kaiming初始化
        self.stride = stride
        self.padding = padding
        self.requires_grad = requires_grad
        self.output_channel = filter_shape[0]   # 输出通道数
        self.input_channel = filter_shape[1]    # 输入通道数
        self.filter_size = filter_shape[2]  # 卷积核大小
        if bias:
            self.bias = parameter(np.random.randn(self.output_channel))
        else:
            self.bias =None

    def forward(self, input):
        """
        :param input:feature map 形状：[N,C,H,W]
        :return:
        """
        # 第一步边缘填充
        if self.padding == "VALID":
            self.x = input
        if self.padding == "SAME":
            p = self.filter_size // 2
            self.x = np.lib.pad(input, ((0,0),(0,0),(p,p),(p,p)), "constant")
        # 第二步处理输入的宽高不能恰好的被卷积核的大小和选定的步长所整除
        x_fit = (self.x.shape[2] - self.filter_size) % self.stride
        y_fit = (self.x.shape[3] - self.filter_size) % self.stride

        if self.stride > 1:
            if x_fit != 0:
                self.x = self.x[:, :, 0:self.x.shape[2] - x_fit, :]
            if y_fit != 0:
                self.x = self.x[:, :, :, 0:self.x.shape[3] - y_fit]

        # 实现卷积
        N, _, H, W = self.x.shape
        O, C, K, K = self.weight.data.shape
        weight_cols = self.weight.data.reshape(O, -1).T
        x_cols = self.img2col(self.x, self.filter_size, self.filter_size, self.stride)
        result = np.dot(x_cols, weight_cols) + self.bias.data
        output_H, output_W = (H-self.filter_size)//self.stride + 1, (W-self.filter_size)//self.stride + 1
        result = result.reshape((N, result.shape[0]//N, -1)).reshape((N, output_H, output_W, O))
        return result.transpose((0, 3, 1, 2))


    def backward(self, eta, lr):
        """
        :param eta:上一层返回的梯度[N,O,output_H, output_W]
        param lr:学习率
        :return:
        """
        # 在eta行行列列之间插入插入0后，计算W,b的梯度；然后进行padding， padding后计算返回上一层的梯度


        # 第一步步长大于1要在行行和列列之间插0
        if self.stride > 1:
            N, O, output_H, output_W = eta.shape
            inserted_H, inserted_W = output_H + (output_H-1)*(self.stride-1), output_W + (output_W-1)*(self.stride-1)
            inserted_eta = np.zeros((N, O, inserted_H, inserted_W))
            inserted_eta[:,:,::self.stride, ::self.stride] = eta
            eta = inserted_eta

        # 计算本层的W,b的梯度
        N, _, output_H, output_W = eta.shape
        self.b_grad = eta.sum(axis=(0,2,3))
        self.W_grad = np.zeros(self.weight.data.shape)      # 形状[O, C, K, K]
        for i in range(self.filter_size):
            for j in range(self.filter_size):
                self.W_grad[:,:,i,j] = np.tensordot(eta, self.x[:,:,i:i+output_H,j:j+output_W], ([0,2,3], [0,2,3]))
        # 权重更新
        self.weight.data -= lr * self.W_grad / N
        self.bias.data -= lr * self.b_grad / N


        # 第二步边缘填充
        if self.padding == "VALID":
            p = self.filter_size - 1
            pad_eta = np.lib.pad(eta, ((0,0),(0,0),(p,p),(p,p)), "constant", constant_values=0)
            eta = pad_eta
        elif self.padding == "SAME":
            p = self.filter_size // 2
            pad_eta = np.lib.pad(eta, ((0, 0), (0, 0), (p, p), (p, p)), "constant", constant_values=0)
            eta = pad_eta

        # 计算传到上一层的梯度
        _, C, _, _ = self.weight.data.shape
        weight_flip = np.flip(self.weight.data, (2,3))  # 卷积核旋转180度
        weight_flip_swap = np.swapaxes(weight_flip, 0, 1)  # 交换输入、输出通道的顺序[C,O,H,W]
        weight_flip = weight_flip_swap.reshape(C, -1).T
        x_cols = self.img2col(eta, self.filter_size, self.filter_size, self.stride)
        result = np.dot(x_cols, weight_flip)
        N, _, H, W = eta.shape
        output_H, output_W = (H - self.filter_size) // self.stride + 1, (W - self.filter_size) // self.stride + 1
        result = result.reshape((N, result.shape[0] // N, -1)).reshape((N, output_H, output_W, C))
        self.weight.grad = result.transpose((0, 3, 1, 2))

        return self.weight.grad


    def img2col(self, x, filter_size_x, filter_size_y, stride):
        """
        现代计算机运算中矩阵运算已经极为成熟（无论是速度还是内存），因此思路是将x中的每个卷积单位[N,C,K,K]展开为行向量，然后与组成二维矩阵
        与展开的权重进行矩阵乘法。最后把结果reshape一下就可以了。
        缺点：虽然提升了速度，但是增大的内存开销（因为x展开成的二维矩阵，存在大量重复元素）
        :param x:输入的feature map形状：[N,C,H,W]
        :param filter_size_x:卷积核的尺寸x
        :param filter_size_y:卷积核的尺寸y
        :param stride:卷积步长
        :return:二维矩阵 形状：[(H-filter_size+1)/stride * (W-filter_size+1)/stride*N, C * filter_size_x * filter_size_y]
        """

        N, C, H, W = x.shape
        output_H, output_W = (H-filter_size_x)//stride + 1, (W-filter_size_y)//stride + 1
        out_size = output_H * output_W
        x_cols = np.zeros((out_size*N, filter_size_x*filter_size_y*C))
        for i in range(0, H-filter_size_x+1, stride):
            i_start = i * output_W
            for j in range(0, W-filter_size_y+1, stride):
                temp = x[:,:, i:i+filter_size_x, j:j+filter_size_y].reshape(N,-1)
                x_cols[i_start+j::out_size, :] = temp
        return x_cols


# ## 3.3池化层

# ### 3.3.1 MaxPooling

# In[ ]:


class Maxpooling():
    def __init__(self, kernel_size=(2, 2), stride=2, ):
        """
        :param kernel_size:池化核的大小(kx,ky)
        :param stride: 步长
        这里有个默认的前提条件就是：kernel_size=stride
        """
        self.ksize = kernel_size
        self.stride = stride

    def forward(self, input):
        """
        :param input:feature map形状[N,C,H,W]
        :return:maxpooling后的结果[N,C,H/ksize,W/ksize]
        """
        N, C, H, W = input.shape
        out = input.reshape(N, C, H//self.stride, self.stride, W//self.stride, self.stride)
        out = out.max(axis=(3,5))
        self.mask = out.repeat(self.ksize[0], axis=2).repeat(self.ksize[1], axis=3) != input
        return out

    def backward(self, eta):
        """
        :param eta:上一层返回的梯度[N,O,H,W]
        :return:
        """
        result = eta.repeat(self.ksize[0], axis=2).repeat(self.ksize[1], axis=3)
        result[self.mask] = 0
        return result


# ### 3.3.2AveragePooling

# In[ ]:


class Averagepooling():
    def __init__(self, kernel_size=(2,2), stride=2):
        self.ksize = kernel_size
        self.stride = stride

    def forward(self, input):
        N, C, H, W = input.shape
        out = input.reshape((N, C, H//self.ksize, self.ksize, W//self.ksize, self.ksize))
        out = out.sum(axis=(3,5))
        out = out / self.ksize**2
        return out

    def backward(self, eta):
        result = eta.repeat(self.ksize, axis=2).repeat(self.ksize, axis=3)
        return result


# ## 3.4 全连接

# In[ ]:


class fc():
    def __init__(self, input_num, output_num, bias=True, requires_grad=True):
        """
        :param input_num:输入神经元个数
        :param output_num: 输出神经元的个数
        """
        self.input_num = input_num          # 输入神经元个数
        self.output_num = output_num        # 输出神经元个数
        self.requires_grad = requires_grad
        self.weight = parameter(np.random.randn(self.input_num, self.output_num) * (2/self.input_num**0.5))
        if bias:
            self.bias = parameter(np.random.randn(self.output_num))
        else:
            self.bias = None


    def forward(self, input):
        """
        :param input: 输入的feature map 形状：[N,C,H,W]或[N,C*H*W]
        :return:
        """
        self.input_shape = input.shape    # 记录输入数据的形状
        if input.ndim > 2:
            N, C, H, W = input.shape
            self.x = input.reshape((N, -1))
        elif input.ndim == 2:
            self.x = input
        else:
            print("fc.forward的输入数据维度存在问题")
        result = np.dot(self.x, self.weight.data)
        if self.bias is not None:
            result = result + self.bias.data
        return result


    def backward(self, eta, lr):
        """
        :param eta:由上一层传入的梯度 形状：[N,output_num]
        :param lr:学习率
        :return: self.weight.grad 回传到上一层的梯度
        """
        N, _ = eta.shape
        # 计算传到下一层的梯度
        next_eta = np.dot(eta, self.weight.data.T)
        self.weight.grad = np.reshape(next_eta, self.input_shape)

        # 计算本层W,b的梯度
        x = self.x.repeat(self.output_num, axis=0).reshape((N, self.output_num, -1))
        self.W_grad = x * eta.reshape((N, -1, 1))
        self.W_grad = np.sum(self.W_grad, axis=0) / N
        self.b_grad = np.sum(eta, axis=0) / N


        # 权重更新
        self.weight.data -= lr * self.W_grad.T
        self.bias.data -= lr * self.b_grad

        return self.weight.grad


# ## 3.5激活函数

# ### 3.5.1 ReLu

# In[ ]:


class Relu():
    def forward(self, x):
        self.x = x
        return np.maximum(self.x, 0)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta


# ### 3.5.2 Sigmoid

# In[ ]:


class sigmoid():
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, eta):
        result = eta * (self.out * (1-self.out))
        return result


# ### 3.5.3 Tanh

# In[ ]:


class tanh():
    def forward(self, x):
        temp1 = np.exp(x) - np.exp(-x)
        temp2 = np.exp(x) + np.exp(-x)
        self.out = temp1 / temp2
        return self.out

    def backward(self, eta):
        return eta * (1 - np.square(self.out))


# ## 3.6归一化层

# ### 3.6.1BatchNorm

# In[ ]:


# BN层有两种实现的方法：一种是原论文中提到的对每个pixel计算均值、方差，并且学习两个参数alpha、beta，但是这样的话参数量较大；
# 所以一般实现采用第二种方法：每个通道计算均值、方差，并学习两个参数alpha、beta。
class BN():
    def __init__(self, channel, moving_decay=0.9, is_train=True):
        """
        :param input_shape:输入需要归一化的数据：[N,C,H,W]
        :param is_train: 当前是否为训练状态
        """
        self.alpha = parameter(np.ones((channel,1,1)))
        self.beta = parameter(np.zeros((channel,1,1)))
        self.is_train = is_train
        self.eps = 1e-5 # 数据归一化时防止下溢

        self.moving_mean = np.zeros((channel,1,1))
        self.moving_var = np.zeros((channel,1,1))
        self.moving_decay = moving_decay


    def forward(self, x, is_train=True):
        """
        :param x:输入的feature map：[N,C,H,W]
        :return: batch_normalization的结果[N,C,H,W]
        """
        self.is_train = is_train
        N, C, H, W = x.shape
        self.x = x
        if N <= 4:
            print("batch size较小，BN层不能准确估计数据集的均值和方差，不建议使用BN层")
            return x

        if self.is_train:       # 判断是否为训练状态
            self.mean = np.mean(x, axis=(0,2,3))[:,np.newaxis, np.newaxis]
            self.var = np.var(x, axis=(0,2,3))[:,np.newaxis, np.newaxis]

            # 计算滑动平均
            if (np.sum(self.moving_mean)==0) and (np.sum(self.moving_var)==0):
                self.moving_mean = self.mean
                self.moving_var = self.var
            else:
                self.moving_mean = self.moving_mean * self.moving_decay + (1-self.moving_decay) * self.mean
                self.moving_var = self.moving_var * self.moving_decay + (1-self.moving_decay) * self.var

            self.y = (x - self.mean) / np.sqrt(self.var + self.eps)
            return  self.alpha.data * self.y + self.beta.data
        else:   # 如果为测试阶段
            self.y = (x - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
            return  self.alpha.data * self.y + self.beta.data


    def backward(self, eta, lr):
        """
        :param eta:
        :param lr:学习率
        :return:
        """
        # 计算alpha和beta的梯度
        N, _, H, W = eta.shape
        alpha_grad = np.sum(eta * self.y, axis=(0,2,3))
        beta_grad = np.sum(eta, axis=(0,2,3))

        # 返回到上一层的梯度
        # 注：这里关于ymean_grad和yvar_grad的计算不确定，欢迎留言指正
        yx_grad = (eta * self.alpha.data)
        ymean_grad = (-1.0 / np.sqrt(self.var +self.eps)) * yx_grad
        ymean_grad = np.sum(ymean_grad, axis=(2,3))[:,:,np.newaxis,np.newaxis] / (H*W)
        yvar_grad = -0.5*yx_grad*(self.x - self.mean) / (self.var+self.eps)**(3.0/2)
        yvar_grad = 2 * (self.x-self.mean) * np.sum(yvar_grad,axis=(2,3))[:,:,np.newaxis,np.newaxis] / (H*W)
        result = yx_grad*(1 / np.sqrt(self.var +self.eps)) + ymean_grad + yvar_grad


        self.alpha.data -= lr * alpha_grad[:,np.newaxis, np.newaxis] / N
        self.beta.data -=  lr * beta_grad[:, np.newaxis, np.newaxis] / N

        return result


# ## 3.7 Dropout

# In[ ]:


class Dropout():
    def __init__(self, drop_rate=0.5, is_train=True):
        """
        :param drop_rate: 随机丢弃神经元的概率
        :param is_train: 当前是否为训练状态
        """
        self.drop_rate = drop_rate
        self.is_train = is_train
        self.fix_value = 1 - drop_rate   # 修正期望，保证输出值的期望不变


    def forward(self, x):
        """
        :param x:[N, m] N为batch_size, m为神经元个数
        :return:
        """
        if self.is_train==False:    # 当前为测试状态
            return x
        else:             # 当前为训练状态
            N, m = x.shape
            self.save_mask = np.random.uniform(0, 1, m) > self.drop_rate   # save_mask中为保留的神经元
            return (x * self.save_mask) / self.fix_value


    def backward(self, eta):
        if self.is_train==False:
            return eta
        else:
            return eta * self.save_mask


# ## 3.8Loss

# In[ ]:


class softmax():
    # def __init__(self, shape):
    #     """
    #     :param shape:[N, m] 其中N为batch_size，m为要预测的类别
    #     """
    #     self.out = np.zeros(shape)
    #     self.eta = np.zeros(shape)

    def calculate_loss(self, x, label):
        """
        :param x: 上一层输出的向量：[N, m] 其中N表示batch，m表示输出节点个数
        :param label:数据的真实标签：[N]
        :return:
        """
        N, _ = x.shape
        self.label = np.zeros_like(x)
        for i in range(self.label.shape[0]):
            self.label[i, label[i]] = 1

        self.x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])   # 为了防止x中出现极值导致溢出，每个样本减去其中最大的值
        sum_x = np.sum(self.x, axis=1)[:, np.newaxis]
        self.prediction = self.x / sum_x

        self.loss = -np.sum(np.log(self.prediction+1e-6) * self.label)  # 防止出现log(0)的情况
        return self.loss / N

    def prediction_func(self, x):
        x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])  # 为了防止x中出现极值导致溢出，每个样本减去其中最大的值
        sum_x = np.sum(x, axis=1)[:, np.newaxis]
        self.out = x / sum_x
        return self.out


    def gradient(self):
        self.eta = self.prediction.copy() - self.label
        return self.eta


# # 4 网络结构--LeNet

# In[ ]:


class LeNet5():
    def __init__(self):
        self.conv1 = conv_fast((6, 1, 5, 5), stride=1, padding='SAME', bias=True, requires_grad=True)
        self.pooling1 = Maxpooling(kernel_size=(2, 2), stride=2)
        self.BN1 = BN(6, moving_decay=0.9, is_train=True)
        self.relu1 = Relu()

        self.conv2 = conv_fast((16, 6, 5, 5), stride=1, padding="VALID", bias=True, requires_grad=True)
        self.pooling2 = Maxpooling(kernel_size=(2, 2), stride=2)
        self.BN2 = BN(16, moving_decay=0.9, is_train=True)
        self.relu2 = Relu()

        self.conv3 = conv_fast((120, 16, 5, 5), stride=1, padding="VALID", bias=True, requires_grad=True)

        self.fc4 = fc(120*1*1, 84, bias=True, requires_grad=True)
        self.relu4 = Relu()
        self.fc5 = fc(84, 10, bias=True, requires_grad=True)

        self.softmax = softmax()

    def forward(self, imgs, labels, is_train=True):
        """
        :param imgs:输入的图片：[N,C,H,W]
        :param labels:
        :return:
        """
        x = self.conv1.forward(imgs)
        x = self.pooling1.forward(x)
        x = self.BN1.forward(x, is_train)
        x = self.relu1.forward(x)

        x = self.conv2.forward(x)
        x = self.pooling2.forward(x)
        x = self.BN2.forward(x, is_train)
        x = self.relu2.forward(x)

        x = self.conv3.forward(x)

        x = self.fc4.forward(x)
        x = self.relu4.forward(x)
        x = self.fc5.forward(x)
        
        prediction = self.softmax.prediction_func(x)
        
        if is_train == True:
            loss = self.softmax.calculate_loss(x, labels)
            return loss, prediction
        else:
            return prediction
            


    def backward(self, lr):
        """
        :param lr:学习率
        :return:
        """
        eta = self.softmax.gradient()

        eta = self.fc5.backward(eta, lr)
        eta = self.relu4.backward(eta)
        eta = self.fc4.backward(eta, lr)

        eta = self.conv3.backward(eta, lr)

        eta = self.relu2.backward(eta)  # 激活层没有参数，不需要学习
        eta = self.BN2.backward(eta, lr)
        eta = self.pooling2.backward(eta)     # 池化层没有参数，不需要学习
        eta = self.conv2.backward(eta, lr)

        eta = self.relu1.backward(eta)
        eta = self.BN1.backward(eta, lr)
        eta = self.pooling1.backward(eta)
        eta = self.conv1.backward(eta, lr)


# # 5训练
# 
# **由于kaggle上cuda使用困难 这里只使用 epoch=3来减少运算时间**

# In[ ]:


import glob
import struct
import time
import matplotlib.pyplot as plt
from functools import reduce
import os 

#超参数
batch_size = 64  # 训练时的batch size
epoch = 3 #由于kaggle上cuda使用困难 这里只使用 epoch=3来减少运算时间
learning_rate = 1e-3

#存储中间过程
x = []  # 保存训练过程中x轴的数据（训练次数）用于画图
y_loss = []  # 保存训练过程中y轴的数据（loss）用于画图
y_acc = []
iterations_num = 0 # 记录训练的迭代次数

#实例化网络
net = LeNet5()


# In[ ]:


for E in range(epoch):
    batch_loss = 0
    batch_acc = 0

    epoch_loss = 0
    epoch_acc = 0

    for i in range(train_imgs.shape[0] // batch_size):
        img = train_imgs[i*batch_size:(i+1)*batch_size].reshape(batch_size, 1, 28, 28)
        img = normalization(img)
        label = train_y[i*batch_size:(i+1)*batch_size]
        loss, prediction = net.forward(img, label, is_train=True)   # 训练阶段

        epoch_loss += loss
        batch_loss += loss
        for j in range(prediction.shape[0]):
            if np.argmax(prediction[j]) == label[j]:
                epoch_acc += 1
                batch_acc += 1
        net.backward(learning_rate)

        if (i+1)%50== 0:
            print("epoch:%2d , batch:%2d , avg_batch_acc:%.4f , avg_batch_loss:%.4f  "
                  % (E+1, i+1, batch_acc/(batch_size*50), batch_loss/(batch_size*50)))
            iterations_num += 1
            x.append(iterations_num)
            y_loss.append(batch_loss/(batch_size*50))
            y_acc.append(batch_acc/(batch_size*50))
            batch_loss = 0
            batch_acc = 0

    print("##epoch:%d , avg_epoch_acc:%.4f , avg_epoch_loss:%.4f ##"% (E+1, epoch_acc/train_imgs.shape[0], epoch_loss/train_imgs.shape[0]))


# In[ ]:


plt.subplot(1, 2, 1)
plt.title('Train-Loss')  # 添加子标题
plt.xlabel('iterations', fontsize=10)  # 添加轴标签
plt.ylabel('Loss', fontsize=10)
plt.plot(x, y_loss, 'r')
plt.subplot(1, 2, 2)
plt.title('Accuracy')  # 添加子标题
plt.xlabel('iterations', fontsize=10)  # 添加轴标签
plt.ylabel('acc', fontsize=10)
plt.plot(x, y_acc, 'b')


# # 6 预测

# In[ ]:


img = test_imgs.reshape(test_imgs.shape[0], 1, 28, 28)
img = normalization(img)
prediction = net.forward(img, labels=None, is_train=False)   # 训练阶段
pre_label = np.argmax(prediction ,axis=1)
print(pre_label)

