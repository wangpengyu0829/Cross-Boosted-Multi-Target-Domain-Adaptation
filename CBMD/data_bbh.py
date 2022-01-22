import os
import cv2
import random
import torch
import numpy as np
import torch.utils.data as data
import math

def is_img1(x):
    if x.endswith('.png') and not(x.startswith('._')):
        return True
    else:
        return False
    
def is_img2(x):
    if x.endswith('.png') and not(x.startswith('._')):
        return True
    else:
        return False
    
def _np2Tensor(img):  
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float() # numpy 转化为 tensor
    return tensor

'''读取训练数据'''
class get_train(data.Dataset):
    def __init__(self, source_path_B, slabel_path_B, source_path_C, slabel_path_C, target_path_B, target_path_C, patch_size):
        self.patch_size = patch_size   
        self.source_path_B = source_path_B 
        self.slabel_path_B = slabel_path_B
        self.source_path_C = source_path_C
        self.slabel_path_C = slabel_path_C
        self.target_path_B = target_path_B
        self.target_path_C = target_path_C
        self.id_to_trainid = {0: 0, 255: 1}
        self._set_filesystem(self.source_path_B, self.slabel_path_B, self.source_path_C, self.slabel_path_C, self.target_path_B, self.target_path_C)   
        self.images_s, self.images_sl, self.images_t, self.images_tl, self.images_t_B, self.images_t_C = self._scan()            
        self.repeat = 2
        
    '''打印路径'''        
    def _set_filesystem(self, dir_s_B, dir_sl_B, dir_s_C, dir_sl_C, dir_t_B, dir_t_C):
        self.dir_s_B = dir_s_B
        self.dir_sl_B = dir_sl_B
        self.dir_s_C = dir_s_C
        self.dir_sl_C = dir_sl_C
        self.dir_t_B = dir_t_B
        self.dir_t_C = dir_t_C
        print('********* Train dir *********')
        print(self.dir_s_B)
        print(self.dir_sl_B)
        print(self.dir_s_C)
        print(self.dir_sl_C)
        print(self.dir_t_B)
        print(self.dir_t_C)
        
    '''遍历图像，获取名称集合'''
    def _scan(self):
        list_s_B = sorted([os.path.join(self.dir_s_B, x) for x in os.listdir(self.dir_s_B) if is_img1(x)])  # 遍历 groudtruth 路径中的图像，其名字形成列表
        random.shuffle(list_s_B)
        list_sl_B = [os.path.splitext(x)[0]+'.png' for x in list_s_B]
        list_sl_B = [os.path.join(self.dir_sl_B, os.path.split(x)[-1]) for x in list_sl_B]  # 根据list_s中的图像名+.png 遍历训练图像路径中的图像，其名字形成列表
        
        list_s_C = [os.path.splitext(x)[0]+'.png' for x in list_s_B]
        list_s_C = [os.path.join(self.dir_s_C, os.path.split(x)[-1]) for x in list_s_C] 
        random.shuffle(list_s_C)
        list_sl_C = [os.path.splitext(x)[0]+'.png' for x in list_s_C]
        list_sl_C = [os.path.join(self.dir_sl_C, os.path.split(x)[-1]) for x in list_sl_C]
        
        list_t_B = [os.path.splitext(x)[0]+'.png' for x in list_s_C]
        list_t_B = [os.path.join(self.dir_t_B, os.path.split(x)[-1]) for x in list_t_B]
        list_t_C = [os.path.splitext(x)[0]+'.png' for x in list_s_C]
        list_t_C = [os.path.join(self.dir_t_C, os.path.split(x)[-1]) for x in list_t_C]

        return list_s_B, list_sl_B, list_s_C, list_sl_C, list_t_B, list_t_C                  

    def __getitem__(self, idx):
        img_s, img_sl, img_t, img_tl, img_s_B, img_s_C, filename_s, filename_sl, filename_t, filename_tl, filename_t_B, filename_t_B = self._load_file(idx)      # 获取图像
        assert img_s.shape[0]==img_sl.shape[0] # 大小相等 # 如果可训练
        assert img_t.shape[0]==img_tl.shape[0] # 大小相等 # 如果可训练
#        xs = random.randint(0, img_s.shape[0] - self.patch_size)           # img_n.shape = (h, w, 3)
#        ys = random.randint(0, img_s.shape[1] - self.patch_size)
##        xt = random.randint(0, img_t.shape[0] - self.patch_size)          # img_n.shape = (h, w, 3)
##        yt = random.randint(0, img_t.shape[1] - self.patch_size)
#        img_s = img_s[xs : xs+self.patch_size, ys : ys+self.patch_size, :] # 随机裁剪一个 patch
#        img_sl = img_sl[xs : xs+self.patch_size, ys : ys+self.patch_size, :]
#        img_t = img_t[xt : xt+self.patch_size, yt : yt+self.patch_size, :]
        img_s  = _np2Tensor(img_s)                                          # 转化为 tensor
        img_sl = _np2Tensor(img_sl)
        img_t  = _np2Tensor(img_t)
        img_tl = _np2Tensor(img_tl)
        img_s_B  = _np2Tensor(img_s_B)
        img_s_C = _np2Tensor(img_s_C)
        return img_s, img_sl, img_t, img_tl, img_s_B, img_s_C

    def __len__(self):
        return len(self.images_s) * self.repeat
        
    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        return idx % len(self.images_s)   # 余数

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)    # 选取 idx
        file_s = self.images_s[idx]   # 选取训练图像名
        file_sl = self.images_sl[idx] # 选取 GT 图像名
        file_t = self.images_t[idx]   # 选取训练图像名
        file_tl = self.images_tl[idx]   # 选取训练图像名
        file_t_B = self.images_t_B[idx]   # 选取目标图像名
        file_t_C = self.images_t_C[idx]   # 选取目标图像名
        
        img_s = cv2.cvtColor(cv2.imread(file_s),   cv2.COLOR_BGR2RGB)    # 读取训练图像
        img_s = cv2.resize(img_s, (256, 256), interpolation=cv2.INTER_CUBIC) 
        if np.max(img_s)>1: img_s = img_s/255.0   # 归一化     
        
        labels = cv2.cvtColor(cv2.imread(file_sl), cv2.COLOR_BGR2GRAY)
        labels = cv2.resize(labels, (256, 256), interpolation=cv2.INTER_NEAREST) 
        labels = cv2.threshold(labels, 128, 255, cv2.THRESH_BINARY)
        labels = labels[1]
#        print(np.unique(labels))
        img_sl = 999*np.ones(labels.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            img_sl[labels == k] = v
        img_sl = np.expand_dims(img_sl, 2)
#        print(file_t)
        img_t = cv2.cvtColor(cv2.imread(file_t),   cv2.COLOR_BGR2RGB)
        img_t = cv2.resize(img_t, (256, 256), interpolation=cv2.INTER_CUBIC)
        if np.max(img_t)>1: img_t = img_t/255.0  

        labelt = cv2.cvtColor(cv2.imread(file_tl), cv2.COLOR_BGR2GRAY)
        labelt = cv2.resize(labelt, (256, 256), interpolation=cv2.INTER_NEAREST) 
        labelt = cv2.threshold(labelt, 128, 255, cv2.THRESH_BINARY)
        labelt = labelt[1]
#        print(np.unique(labels))
        img_tl = 999*np.ones(labelt.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            img_tl[labelt == k] = v
        img_tl = np.expand_dims(img_tl, 2)
        
        img_t_B = cv2.cvtColor(cv2.imread(file_t_B),   cv2.COLOR_BGR2RGB)    # 读取训练图像
        img_t_B = cv2.resize(img_t_B, (256, 256), interpolation=cv2.INTER_CUBIC) 
        if np.max(img_t_B)>1: img_t_B = img_t_B/255.0   # 归一化  
        
        img_t_C = cv2.cvtColor(cv2.imread(file_t_C),   cv2.COLOR_BGR2RGB)    # 读取训练图像
        img_t_C = cv2.resize(img_t_C, (256, 256), interpolation=cv2.INTER_CUBIC) 
        if np.max(img_t_C)>1: img_t_C = img_t_C/255.0   # 归一化  
        
        filename_s = os.path.splitext(os.path.split(file_s)[-1])[0]   # 训练图像每个图像的图像名
        filename_sl = os.path.splitext(os.path.split(file_sl)[-1])[0]   # GT图像每个图像的图像名
        filename_t = os.path.splitext(os.path.split(file_t)[-1])[0]   # GT图像每个图像的图像名
        filename_tl = os.path.splitext(os.path.split(file_tl)[-1])[0]   # GT图像每个图像的图像名
        filename_t_B = os.path.splitext(os.path.split(file_t_B)[-1])[0]   # GT图像每个图像的图像名
        filename_t_C = os.path.splitext(os.path.split(file_t_C)[-1])[0]   # GT图像每个图像的图像名
        return img_s, img_sl, img_t, img_tl, img_t_B, img_t_C, filename_s, filename_sl, filename_t, filename_tl,filename_t_B, filename_t_C  # 输出图像和图像名
    
    
'''读取训练数据'''
class get_test(data.Dataset):
    def __init__(self, source_path, slabel_path, target_path, tlabel_path):
        self.source_path = source_path 
        self.slabel_path = slabel_path
        self.target_path = target_path
        self.tlabel_path = tlabel_path
        self.id_to_trainid = {0: 0, 255: 1}
        self._set_filesystem(self.source_path, self.slabel_path, self.target_path, self.tlabel_path)   
        self.images_s, self.images_sl, self.images_t, self.images_tl = self._scan()            
        
    '''打印路径'''        
    def _set_filesystem(self, dir_s, dir_sl, dir_t, dir_tl):
        self.dir_s = dir_s
        self.dir_sl = dir_sl
        self.dir_t = dir_t
        self.dir_tl = dir_tl
        print('********* Train dir *********')
        print(self.dir_s)
        print(self.dir_sl)
        print(self.dir_t)
        print(self.dir_tl)
        
    '''遍历图像，获取名称集合'''
    def _scan(self):
        list_s = sorted([os.path.join(self.dir_s, x) for x in os.listdir(self.dir_s) if is_img1(x)])  # 遍历 groudtruth 路径中的图像，其名字形成列表
        list_sl = [os.path.splitext(x)[0]+'.png' for x in list_s]
        list_sl = [os.path.join(self.dir_sl, os.path.split(x)[-1]) for x in list_sl]  # 根据list_c中的图像名+.png 遍历训练图像路径中的图像，其名字形成列表
        list_t = [os.path.splitext(x)[0]+'.png' for x in list_s]
        list_t = [os.path.join(self.dir_t, os.path.split(x)[-1]) for x in list_t]  
        list_tl = [os.path.splitext(x)[0]+'.png' for x in list_t]
        list_tl = [os.path.join(self.dir_tl, os.path.split(x)[-1]) for x in list_tl]
        return list_s, list_sl, list_t, list_tl                  

    def __getitem__(self, idx):
        img_s, img_sl, img_t, img_tl, filename_s, filename_sl, filename_t, filename_tl = self._load_file(idx)      # 获取图像
        assert img_s.shape[0]==img_sl.shape[0] # 大小相等 # 如果可训练
        assert img_t.shape[0]==img_tl.shape[0] # 大小相等 # 如果可训练
        img_s = _np2Tensor(img_s)                                        # 转化为 tensor
        img_sl = _np2Tensor(img_sl)
        img_t = _np2Tensor(img_t)
        img_tl = _np2Tensor(img_tl)
        return img_s, img_sl, img_t, img_tl

    def __len__(self):
        return len(self.images_s)
        
    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        return idx   # 余数

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)    # 选取 idx
        file_s = self.images_s[idx]   # 选取训练图像名
        file_sl = self.images_sl[idx] # 选取 GT 图像名
        file_t = self.images_t[idx]   # 选取训练图像名
        file_tl = self.images_tl[idx]   # 选取训练图像名
        
        img_s = cv2.cvtColor(cv2.imread(file_s), cv2.COLOR_BGR2RGB)    # 读取训练图像
        img_s = cv2.resize(img_s, (256, 256), interpolation=cv2.INTER_CUBIC) 
        if np.max(img_s)>1: img_s = img_s/255.0   # 归一化
        
        labels = cv2.cvtColor(cv2.imread(file_sl), cv2.COLOR_BGR2GRAY)
        labels = cv2.resize(labels, (256, 256), interpolation=cv2.INTER_NEAREST) 
        labels = cv2.threshold(labels, 128, 255, cv2.THRESH_BINARY)
        labels = labels[1]
        img_sl = 999*np.ones(labels.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            img_sl[labels == k] = v
        img_sl = np.expand_dims(img_sl, 2)

        img_t = cv2.cvtColor(cv2.imread(file_t), cv2.COLOR_BGR2RGB)
        img_t = cv2.resize(img_t, (256, 256), interpolation=cv2.INTER_CUBIC) 
        if np.max(img_t)>1: img_t = img_t/255.0  
        
        labelt = cv2.cvtColor(cv2.imread(file_tl), cv2.COLOR_BGR2GRAY)
        labelt = cv2.resize(labelt, (256, 256), interpolation=cv2.INTER_NEAREST)
        labelt = cv2.threshold(labelt, 128, 255, cv2.THRESH_BINARY)
        labelt = labelt[1]
        img_tl = 999*np.ones(labelt.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            img_tl[labelt == k] = v
        img_tl = np.expand_dims(img_tl, 2)
        
        filename_s = os.path.splitext(os.path.split(file_s)[-1])[0]   # 训练图像每个图像的图像名
        filename_sl = os.path.splitext(os.path.split(file_sl)[-1])[0]   # GT图像每个图像的图像名
        filename_t = os.path.splitext(os.path.split(file_t)[-1])[0]   # GT图像每个图像的图像名
        filename_tl = os.path.splitext(os.path.split(file_tl)[-1])[0]   # GT图像每个图像的图像名
        return img_s, img_sl, img_t, img_tl, filename_s, filename_sl, filename_t, filename_tl,  # 输出图像和图像名  
    
    
'''读取训练数据'''
class get_source(data.Dataset):
    def __init__(self, target_path_B, target_path_C):
        self.target_path_B = target_path_B 
        self.target_path_C  = target_path_C 
        self.id_to_trainid = {0: 0, 255: 1}
        self._set_filesystem(self.target_path_B, self.target_path_C)   # 训练图像路径 / groudtruth 路径
        self.images_s, self.images_l = self._scan()             # 获得训练和 GT 图像
        
    '''打印路径'''        
    def _set_filesystem(self, dirt_t_B, dir_t_C):
        self.dirt_t_B = dirt_t_B
        self.dir_t_C = dir_t_C
        print('********* Test dir *********')
        print(self.dirt_t_B)
        print(self.dir_t_C)
        
    '''遍历图像，获取名称集合'''
    def _scan(self):
        list_t_B = sorted([os.path.join(self.dirt_t_B, x) for x in os.listdir(self.dirt_t_B) if is_img2(x)])  # 遍历 groudtruth 路径中的图像，其名字形成列表
        list_t_C = [os.path.splitext(x)[0]+'.png' for x in list_t_B]
        list_t_C = [os.path.join(self.dir_t_C, os.path.split(x)[-1]) for x in list_t_C]  # 根据list_c中的图像名.png 遍历训练图像路径中的图像，其名字形成列表
        return list_t_B, list_t_C                  

    def __getitem__(self, idx):
        img_s, img_l, filename_s, filename_l = self._load_file(idx)  # 获取图像
        assert img_s.shape[0]==img_l.shape[0] # 大小相等 # 如果可训练
        assert img_s.shape[1]==img_l.shape[1] # 大小相等 # 如果可训练
        img_s = _np2Tensor(img_s)       # 转化为 tensor
        img_l = _np2Tensor(img_l)
        return img_s, img_l

    def __len__(self):
        return len(self.images_s)
        
    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        return idx   # 余数

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)    # 选取 idx
        file_s = self.images_s[idx]   # 选取训练图像名
        file_l = self.images_l[idx]   # 选取 GT 图像名
        
        img_s = cv2.cvtColor(cv2.imread(file_s), cv2.COLOR_BGR2RGB)    # 读取训练图像
        img_s = cv2.resize(img_s, (256, 256), interpolation=cv2.INTER_CUBIC) 
        if np.max(img_s)>1: img_s = img_s/255.0   # 归一化
        
        img_l = cv2.cvtColor(cv2.imread(file_l), cv2.COLOR_BGR2RGB)    # 读取训练图像
        img_l = cv2.resize(img_l, (256, 256), interpolation=cv2.INTER_CUBIC) 
        if np.max(img_l)>1: img_l = img_l/255.0   # 归一化

        filename_s = os.path.splitext(os.path.split(file_s)[-1])[0]   # 训练图像每个图像的图像名
        filename_l = os.path.splitext(os.path.split(file_l)[-1])[0]   # GT图像每个图像的图像名
        return img_s, img_l, filename_s, filename_l                   # 输出图像和图像名

    
'''读取训练数据'''
class get_target(data.Dataset):
    def __init__(self, target_path):
        self.target_path = target_path 
        self._set_filesystem(self.target_path)   # 训练图像路径 / groudtruth 路径
        self.images_t = self._scan()             # 获得训练和 GT 图像
        
    '''打印路径'''        
    def _set_filesystem(self, dir_t):
        self.dir_t = dir_t
        print('********* Test dir *********')
        print(self.dir_t)
        
    '''遍历图像，获取名称集合'''
    def _scan(self):
        list_t = sorted([os.path.join(self.dir_t, x) for x in os.listdir(self.dir_t) if is_img2(x)])
        return list_t                  

    def __getitem__(self, idx):
        img_t, filename_t = self._load_file(idx)  # 获取图像
        img_t = _np2Tensor(img_t)       # 转化为 tensor
        return img_t

    def __len__(self):
        return len(self.images_t)
        
    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        return idx   # 余数

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)    # 选取 idx
        file_t = self.images_t[idx]   # 选取训练图像名
        
        img_t = cv2.cvtColor(cv2.imread(file_t), cv2.COLOR_BGR2RGB)    # 读取训练图像
        if np.max(img_t)>1: img_t = img_t/255.0   # 归一化
        xt = math.floor(img_t.shape[0]/32)*32
        yt = math.floor(img_t.shape[1]/32)*32
        img_t = cv2.resize(img_t, (yt, xt), interpolation=cv2.INTER_CUBIC)   
        
        filename_t = os.path.splitext(os.path.split(file_t)[-1])[0]   # 训练图像每个图像的图像名
        return img_t, filename_t      
    
