import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support

def calculate_metrics(true_label,pred_label):
    accuracy = np.mean(true_label == pred_label)
    precision,recall,f1_score,_ = precision_recall_fscore_support(true_label,pred_label,average='weighted')
    kappa = cohen_kappa_score(true_label,pred_label)
    conf_mat = confusion_matrix(true_label,pred_label)

    return accuracy,precision,recall,f1_score,kappa,conf_mat

def extract_subimages(img0,img1,img_size_x = 512,img_size_y = 512,gap = 256,zero_ratio = 0.5):
    cx,cy = np.meshgrid(range(0,img0.shape[0]-img_size_x,gap),range(0,img0.shape[1]-img_size_y,gap))

    res1 = []
    res2 = []
    x_coords = []
    y_coords = []

    for x,y in zip(cx.flatten(),cy.flatten()):
        cur_img = img0[x:x+img_size_x,y:y+img_size_y]
        cur_img2 = img1[x:x+img_size_x,y:y+img_size_y]

        zr = np.mean(np.abs(cur_img)<1e-6)
        if zr > zero_ratio:
        # gap = 100
        # cnt = Counter(cur_img.flatten()[::gap]).most_common()[0][1]
        # if cnt > int(img_size_x*img_size_y*0.5/gap):
            pass
        else:
          # print(x,y,cur_img.shape)
            res1 += [cur_img]
            res2 += [cur_img2]
            x_coords += [x]
            y_coords += [y]

    r1 = np.array(res1).astype('float32')
    r2 = np.array(res2)
    r3 = (x_coords,y_coords)
    return r1,r2,r3

def extract_subimages_generator(img0,img1,batch_size = 16,img_size_x = 512,img_size_y = 512,gap = 256, zero_ratio = 0.5):
    while True:
        cx,cy = np.meshgrid(range(0,img0.shape[0]-img_size_x,gap),range(0,img0.shape[1]-img_size_y,gap))

        res1 = []
        res2 = []
        x_coords = []
        y_coords = []

        
        for x,y in zip(cx.flatten(),cy.flatten()):
            cur_img = img0[x:x+img_size_x,y:y+img_size_y,:]
            cur_img2 = img1[x:x+img_size_x,y:y+img_size_y]

            zr = np.mean(np.abs(cur_img[:,:,0])<1e-6)
            if zr > zero_ratio:
            # gap = 100
            # cnt = Counter(cur_img.flatten()[::gap]).most_common()[0][1]
            # if cnt > int(img_size_x*img_size_y*0.5/gap):
                pass
            else:
              # print(x,y,cur_img.shape)
                res1 += [cur_img]
                res2 += [cur_img2]
                print(len(res1))
    #             x_coords += [x]
    #             y_coords += [y]
            
            if len(res1) == batch_size:
                print('here lies the problem')
                r1 = np.array(res1).astype('float32')
                r2 = np.array(res2).astype('float32')
#                 r2 = (np.array(res2) == cl)
    #             r3 = (x_coords,y_coords)

#                 plt.imshow(r2[0,...])
#                 plt.show()
                yield r1,r2
                res1 = []
                res2 = []
                x_coords = []
                y_coords = []
#             print(len(cx),len(cy))
        r1 = np.array(res1).astype('float32')
        r2 = np.array(res2).astype('float32')

        yield r1,r2


def extract_random_subimages_generator_multi(img0,img1,n=10,img_size_x = 512,img_size_y = 512, Augmentation=True,zero_ratio = 0.5, num_classes = 5):
    while True:
        n_cur = n
        if Augmentation:
            n_cur = n//4
        if n == 0:
            n_cur = 1
            
        res1 = []
        res2 = []
        i=0
        while i<n_cur:
            x = np.random.randint(0,img0.shape[0]-img_size_x,1)[0]
            y = np.random.randint(0,img0.shape[1]-img_size_y,1)[0]
            cur_img = img0[x:x+img_size_x,y:y+img_size_y,:]
            cur_img2 = img1[x:x+img_size_x,y:y+img_size_y]

            zr = np.mean(np.abs(cur_img[:,:,0])<1e-6)
            if zr > zero_ratio:
        # gap = 100
        # cnt = Counter(cur_img.flatten()[::gap]).most_common()[0][1]
        # if cnt > int(img_size_x*img_size_y*0.5/gap):
              pass
            else:
                res1 += [cur_img]
                res2 += [cur_img2]
                i+=1
        r1 = np.array(res1).astype('float32')
        r2 = np.array(res2)
        r2 = tf.keras.utils.to_categorical(r2,num_classes = num_classes)
        
#         Data Augmentation
        if Augmentation:
            r1 = np.concatenate([r1,r1[:,::-1,:,:],r1[:,:,::-1,:],r1[:,::-1,::-1,:]],axis=0)
            r2 = np.concatenate([r2,r2[:,::-1,:],  r2[:,:,::-1],  r2[:,::-1,::-1]  ],axis=0)

        yield r1,r2

def reconstruct_image(label_img,coords,img_size):
    minx, maxx = min(coords[0]), max(coords[0])
    miny, maxy = min(coords[0]), max(coords[1])

    canvas = np.zeros((maxx-minx+img_size,maxy-miny+img_size))

    for patch_label,coordx, coordy in tqdm(zip(label_img,coords[0],coords[1])):
        canvas[coordx:coordx+img_size,coordy:coordy+img_size] = patch_label
    return canvas

