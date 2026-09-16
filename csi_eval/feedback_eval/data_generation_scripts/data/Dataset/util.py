'''
Copyright (C) 2021. Huawei Technologies Co., Ltd.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
'''

import os
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import copy

class Recoder(object):    
    def __init__(self):
        self.last = 0
        self.values = []
        self.nums = []
    def update(self, val, n=1):
        self.last = val
        self.values.append(val)
        self.nums.append(n)        
    def avg(self):
        sum = np.sum(np.asarray(self.values)*np.asarray(self.nums))
        count = np.sum(np.asarray(self.nums))
        return sum/count
        
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def toNP(tensor):
    return tensor.detach().cpu().numpy()
    
def makeDIRs(folder):
    if not os.path.exists(f'models/{folder}/'):
        os.makedirs(f'models/{folder}/')
    if not os.path.exists(f'results/{folder}/'):
        os.makedirs(f'results/{folder}/')
        
def checkPoint(runCase,epoch,epochs,model,Train,Valid,saveModelInterval,saveLossInterval):    
    if epoch%saveModelInterval==0 or epoch==epochs:
        torch.save(model.state_dict(), f'models/{runCase}/DoraNet_'+str(epoch)+'.pth')
    if epoch%saveLossInterval==0 or epoch==epochs:
        np.save(f'results/{runCase}/Train.npy',np.asarray(Train))
        np.save(f'results/{runCase}/Valid.npy',np.asarray(Valid))
        
def readEnvLink(folder,path,carrierFreq,imageSize,scenario):
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    ])    
    sub_path = os.path.join('../../', folder, path)    
    linklocA_path = os.path.join(sub_path, 'Info.npy')
    linklocA = np.load(linklocA_path, allow_pickle=True, encoding='latin1')
    linklocA[:2] = np.floor(linklocA[:2])    
    env_path = os.path.join(sub_path, 'environment.png')
    env = Image.open(env_path, mode='r')
    ImgSize = linklocA[:2].astype(np.int)
    if ImgSize[0]>ImgSize[1]:
        top = -(ImgSize[0]-ImgSize[1])//2
        left = 0            
    else:
        left = -(ImgSize[1]-ImgSize[0])//2
        top = 0
    env = env.resize(ImgSize)
    envNew = Image.new(env.mode, (np.max(ImgSize), np.max(ImgSize)), (255, 255, 255))  
    envNew.paste(env, (int(-left),int(-top)))
    env.close()
    envNew = 1 - transform(envNew)
    if torch.max(envNew)==0:
        pass
    else:
        envNew = envNew/torch.max(envNew)
    envNew = transforms.functional.resize(envNew,imageSize)
    linklocA[2::2] = (linklocA[2::2]-left)/np.max(ImgSize)
    linklocA[3::2] = (linklocA[3::2]-(ImgSize[1]-top-np.max(ImgSize)))/np.max(ImgSize)    
    P = np.load(os.path.join(sub_path, 'Path.npy'), allow_pickle=True, encoding='latin1')
    Pitem = P.item()
    H = np.load(os.path.join(sub_path, f'H_{carrierFreq}_G.npy'),allow_pickle=True,encoding='latin1')
    Hitem = H.item()    
    sights = []
    distances = []
    gains = []
    angles = []
    if scenario==1:
        bsMax = 5
        ueMax = 30
    else:
        bsMax = 1
        ueMax = 10000
    for bsIdx in range(bsMax):
        for ueIdx in range(ueMax):
            Plink = Pitem[f'bs{bsIdx}_ue{ueIdx}'] if scenario==1 else Pitem[f'bs{bsIdx}_ue{ueIdx:05d}']
            Hlink = Hitem[f'bs{bsIdx}_ue{ueIdx}'] if scenario==1 else Hitem[f'bs{bsIdx}_ue{ueIdx:05d}']            
            tau = Plink['taud']
            firstPath = np.argmin(tau)
            doa = Plink['doa']
            dod = Plink['dod']
            phiDiff = dod[firstPath,1]-doa[firstPath,1]
            if scenario==1:
                BSloc=linklocA[2:].reshape(150,4)[bsIdx*30,:2]
                UEloc=linklocA[2:].reshape(150,4)[ueIdx,2:4]
            else:
                BSloc=linklocA[2:4]
                UEloc=linklocA[4:].reshape(10000,2)[ueIdx,:]
            dis1 = ((UEloc[1]-BSloc[1])**2+(UEloc[0]-BSloc[0])**2+((6-1.5)*2/np.max(ImgSize))**2)**0.5*0.5*np.max(ImgSize)
            dis2 = np.min(tau)*0.3
            distances.append(dis2*2/np.max(ImgSize))
            if np.round(phiDiff,5)==np.round(np.pi,5) and np.round(dis1,5)==np.round(dis2,5):
                sight = 1 # line of sight
            else:
                sight = 0 # non line of sight
            gains.append(10*np.log10(np.sum(np.abs(Hlink)**2)))
            sights.append(sight)
            angles.append([dod[firstPath,1],doa[firstPath,1]])
    return envNew,linklocA,np.asarray(sights),np.asarray(distances),np.asarray(angles),np.asarray(gains)
    
def readChannel(readImage,path,curEnv,bsNo,ueNo,scenario):
    if readImage:
        channelPath = f'../../{path}/{curEnv}/image/bs{bsNo}_' + (f'ue{ueNo}.png' if scenario==1 else f'ue{ueNo:05d}.png')
        H = Image.open(channelPath)
        H = np.asarray(H,dtype=float)
        H = (H[:,:,0]/255+1j*H[:,:,1]/255-0.5-1j*0.5)*2
    else:
        channelPath = f'../../{path}/{curEnv}/array/bs{bsNo}_' + (f'ue{ueNo}.npy' if scenario==1 else f'ue{ueNo:05d}.npy')
        H = np.load(channelPath)
        H = H[0,:,:]+1j*H[1,:,:]
    return H

def antennaPosition(N,spacing,Basis):
    N0,N1,N2 = N
    p0 = spacing[0]*np.linspace(-(N0-1)*0.5,(N0-1)*0.5,N0)[None,:]*Basis[:,0:1]
    p1 = spacing[1]*np.linspace(-(N1-1)*0.5,(N1-1)*0.5,N1)[None,:]*Basis[:,1:2]
    p2 = spacing[2]*np.linspace(-(N2-1)*0.5,(N2-1)*0.5,N2)[None,:]*Basis[:,2:3]
    p = p0[:,:,None,None] + p1[:,None,:,None] + p2[:,None,None,:]
    position = p.reshape((3, np.prod(N)))
    return position
    
def arrayResponse(angle,position,sortedPath):
    rx = np.sin(angle[sortedPath,0])*np.cos(angle[sortedPath,1])
    ry = np.sin(angle[sortedPath,0])*np.sin(angle[sortedPath,1])
    rz = np.cos(angle[sortedPath,0])
    r  = np.concatenate((rx[:,None],ry[:,None],rz[:,None]),axis=-1)
    r = np.dot(r,position)
    response = np.exp(1j*2*np.pi*r)
    return response
    
def backward(optimizer,loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()            