import os
import h5py
import torch
import numpy as np
import torchvision.transforms as T

import shutil
def normolize(IR):
    negative_pos = torch.where(IR<0)
    IR[negative_pos] = 0
    IR = IR/torch.max(IR)
    return IR

def data_plot(label):
    to_image_transforms = T.ToPILImage()
    img = (label/6*255).to(torch.uint8)
    img = to_image_transforms(img)
    
    img.save("label.jpg")
    pos = { 0 : torch.where(label==0),
            1 : torch.where(label==1),
            2 : torch.where(label==2),
            3 : torch.where(label==3),
            4 : torch.where(label==4),
            5 : torch.where(label==5),
            6 : torch.where(label==6)}
    for i in range(7):
        label = label*0
        label[pos[i]] = 255
        img = to_image_transforms(label.to(torch.uint8))
        img.save(f"label{i+1}.jpg")  
    return

def save_data(IR, label, patches, img_size, folder):
    for i, pt in enumerate(patches):
        IR_patch = IR[:,pt[1]:pt[1]+img_size,pt[0]:pt[0]+img_size]
        Label_patch = label[pt[1]:pt[1]+img_size,pt[0]:pt[0]+img_size]
        np.save(os.path.join(folder, 'IR', f'IR_{i}'),np.array(IR_patch))
        np.save(os.path.join(folder, 'label', f'label_{i}'),np.array(Label_patch))
        
        to_image_transforms = T.ToPILImage()
        IR_patch = (IR_patch[0,:,:]/torch.max(IR_patch[0,:,:])*255).to(torch.uint8)
        Label_patch = (Label_patch/6*255).to(torch.uint8)
        
        img = to_image_transforms(IR_patch)
        img.save(os.path.join(folder, 'IR', f'IR_{i}.jpeg'))
        img = to_image_transforms(Label_patch)
        img.save(os.path.join(folder, 'label', f'label_{i}.jpeg'))
        
def create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(os.path.join(path,'IR'))
    os.mkdir(os.path.join(path,'label'))
            
if __name__ == "__main__":
    data_dir = './'
    train_folder = './train'
    test_folder = './val'
    create_folder(train_folder)
    create_folder(test_folder)
        
        
    IR = torch.from_numpy(np.array(h5py.File(os.path.join(data_dir,'IR.mat'), 'r')['X']))
    IR = normolize(IR)  
    label = torch.from_numpy(np.array(h5py.File(os.path.join(data_dir,'Class.mat'), 'r')['CL']))  # [H, W]
    img_size = 250
    train_test_split = 0.8
    with open("cell_centers.txt", "r") as f:
        pts = f.readlines()
        pts = [pt.split() for pt in pts]
        patches = []
        for pt in pts:
            for i in range(4):
                patches.append([float(pt[0]) - img_size*(i%2), float(pt[1]) - img_size*(i//2)])
        # shuffle
        patches = torch.Tensor(patches).to(torch.int32)
        patches=patches[torch.randperm(len(patches))].view(patches.size())
        patches_train = patches[:int(len(patches)*train_test_split)]
        patches_test = patches[int(len(patches)*train_test_split):]
        
        save_data(IR,label, patches_train, img_size, train_folder)
        save_data(IR,label, patches_test, img_size, test_folder)
    
    data_plot(label)
    
