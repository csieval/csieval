'''
Copyright (C) 2021. Huawei Technologies Co., Ltd.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
'''

import os
from util import *
from tqdm import tqdm
from parameters import *
from torch.utils.data import Dataset,DataLoader

class DoraSet(Dataset):
    def __init__(self):        
        self.fcGHz = float(carrierFreq.replace('_','.'))
        self.fGHz  = np.linspace(-0.5*BWGHz,0.5*BWGHz,sampledCarriers)+self.fcGHz
        self.Hs = {}
        self.Ps = {}
        self.envs = []
        path_list = os.listdir(scenarioFolder)
        path_list.sort(key=lambda x:int(x))
        for i in tqdm(path_list[:ENVnum]):
            self.envs.append(i)
            envPath = os.path.join(scenarioFolder, i)
            H = np.load(os.path.join(envPath, f'H_{carrierFreq}_G.npy'),allow_pickle=True,encoding='latin1')
            self.Hs[len(self.envs)-1] = H.item()
            P = np.load(os.path.join(envPath, 'Path.npy'),allow_pickle=True,encoding='latin1')
            self.Ps[len(self.envs)-1] = P.item()
            if saveAsArray:
                arrayFolder = os.path.join(generatedFolder, i, 'array')
                if not os.path.exists(arrayFolder):
                    os.makedirs(arrayFolder)
            if saveAsImage:
                imageFolder = os.path.join(generatedFolder, i, 'image')
                if not os.path.exists(imageFolder):
                    os.makedirs(imageFolder)
        
    def __getitem__(self, idx):
        curEnv = idx//(BSnum*UEnum)
        curLink = idx%(BSnum*UEnum)
        bsIdx = curLink//UEnum
        ueIdx = curLink%UEnum
        linkStr = f'bs{BSlist[bsIdx]}_' + (f'ue{UElist[ueIdx]}' if scenario==1 else f'ue{UElist[ueIdx]:05d}')
        H = self.Hs[curEnv][linkStr]
        P = self.Ps[curEnv][linkStr]
        tau = np.asarray(P['taud'])
        sortedPath = np.argsort(tau)[:maxPathNum]
        doa = np.asarray(P['doa'])
        dod = np.asarray(P['dod'])
        pos_r = antennaPosition(Nr,spacing_r,Basis_r)        
        res_r = arrayResponse(doa,pos_r,sortedPath)
        pos_t = antennaPosition(Nt,spacing_t,Basis_t)
        res_t = arrayResponse(dod,pos_t,sortedPath)
        pow_t = 10**(Pattern_t['Power']/10)
        norm_H = H*(pow_t**0.5)/(subcarriers**0.5)
        ofdm_H  = norm_H[sortedPath,None]*np.exp(-2*1j*np.pi*tau[sortedPath,None]*self.fGHz[None,:])        
        CFR = np.sum(ofdm_H[:,None,None,:]*res_t[:,:,None,None]*res_r[:,None,:,None],axis=0) # dimensions in (Nt,Nr,Nf)        
        channel = np.reshape(CFR,(-1,sampledCarriers)) # dimensions in (Nt*Nr,Nf)
        if saveAsArray:
            arrayPath = os.path.join(generatedFolder, self.envs[curEnv], 'array', linkStr+'.npy')
            array = np.concatenate((np.real(channel)[None,...],np.imag(channel)[None,...]),axis=0)
            np.save(arrayPath,array)            
        if saveAsImage:
            norm_channel = channel/np.max(np.abs(channel)) # to save as image, should be normalized first
            image = (norm_channel/2+0.5+0.5*1j)*255
            image = np.concatenate((np.uint8(np.round(np.real(image)))[:,:,None],np.uint8(np.round(np.imag(image)))[:,:,None]),axis=-1)
            image = Image.fromarray(image,mode='LA')            
            imagePath = os.path.join(generatedFolder, self.envs[curEnv], 'image', linkStr+'.png')
            image.save(imagePath)
        return channel
        
    def __len__(self):
        return ENVnum*BSnum*UEnum
        
if __name__ == "__main__":
    print('Init...')    
    doraset = DoraSet()
    generator = DataLoader(doraset, batch_size=100, shuffle=False, num_workers=numCores, pin_memory=True)
    print('Generating...')
    for channel in tqdm(generator):
        pass
    print('Done!')