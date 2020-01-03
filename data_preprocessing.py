
#!/usr/bin/python

#reading txt file having all video names

import cv2
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from glob import glob
import matplotlib as mp
import os
import matplotlib.image as img
              
import os
import sys


#Remember to add this path
path = '/work1/s193416/videos/Bredgrund Videos/'
file_finder_dict = defaultdict(dict)


directories = []
for root, dirs, files in os.walk(path):
    for directory in dirs:
            directories.append(directory)
                

def create_file_finder(path):        
    file_finder_dict = defaultdict(dict) 
    
    for directory in directories:
        count = 0
        #print(path + directory +'/')
        for root, dirs, files in os.walk(path + directory + '/'):
            for file in files:
                
                if file.endswith('LRV'):
                    count += 1
                
                    file_finder_dict[directory][count] = file
    return file_finder_dict

file_finder_dict = create_file_finder(path)
       


def create_subfolders(path):
    for directory in file_finder_dict:

        if not os.path.exists(path + 'experiment/' + directory):
            os.makedirs(path + 'experiment/' + directory)
#create_subfolders(path)
    

              

def generate_data(file_finder_dict, path):
  
    
    for directory in file_finder_dict:
        #dir_count += 1
        count = 0
        
        print('Writing imgs for:  ' ,  directory , 'with n. of vids: ', len(directory))
        
        for i in range (1, len(directory)):
            

            
            #Later, you would then loop teh files in file finder to make sure all
            #videos are extracted
            
            
            vidcap = cv2.VideoCapture(path + directory + '/' + file_finder_dict[directory][i])
            #print('lol')
            success, image = vidcap.read()
            #print(success)
            
            while success:
                #print('lol2')
                
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*2000))
                success, image = vidcap.read()
                #print(success)
                if success:
                    #print('writing')
                    cv2.imwrite(path + 'experiment/' + directory + '/' + "video{0}_frame{1}.jpg" .format(i, count), image)
                count = count + 1


def create_file_finder(path):        
    file_finder_dict = defaultdict(dict) 
    
    for directory in directories:
        count = 0
        #print(path + directory +'/')
        for root, dirs, files in os.walk(path + directory + '/'):
            for file in files:
                
                if file.endswith('LRV'):
                    count += 1
                
                    file_finder_dict[directory][count] = file
                    
    return file_finder_dict

def generate_sequences():
    
    for directory in file_finder_dict:
        #dir_count += 1
        count = 0
        
        print('Writing imgs for:  ' ,  directory , 'with n. of vids: ', len(directory))
        
        for i in range (1, len(directory)):
            
            
    
generate_data(file_finder_dict, path)

