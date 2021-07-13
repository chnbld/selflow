# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import os
import sys
import time
import cv2
import glob

from six.moves import xrange
from scipy import misc, io
from tensorflow.contrib import slim

import matplotlib.pyplot as plt
from network import pyramid_processing
from datasets import BasicDataset
from utils import average_gradients, lrelu, occlusion, rgb_bgr
from data_augmentation import flow_resize
from flowlib import flow_to_color, write_flo
from warp import tf_warp

class SelFlowModel(object):
    def __init__(self, batch_size=8, iter_steps=1000000, initial_learning_rate=1e-4, decay_steps=2e5, 
                 decay_rate=0.5, is_scale=True, num_input_threads=4, buffer_size=5000,
                 beta1=0.9, num_gpus=1, save_checkpoint_interval=5000, write_summary_interval=200,
                 display_log_interval=50, allow_soft_placement=True, log_device_placement=False, 
                 regularizer_scale=1e-4, cpu_device='/cpu:0', save_dir='KITTI', checkpoint_dir='checkpoints', 
                 model_name='model', sample_dir='sample', summary_dir='summary', training_mode="no_distillation", 
                 is_restore_model=False, restore_model='./models/KITTI/no_census_no_occlusion',
                 dataset_config={}, self_supervision_config={}):
        self.batch_size = batch_size
        self.iter_steps = iter_steps
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_scale = is_scale
        self.num_input_threads = num_input_threads
        self.buffer_size = buffer_size
        self.beta1 = beta1       
        self.num_gpus = num_gpus
        self.save_checkpoint_interval = save_checkpoint_interval
        self.write_summary_interval = write_summary_interval
        self.display_log_interval = display_log_interval
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.regularizer_scale = regularizer_scale
        self.training_mode = training_mode
        self.is_restore_model = is_restore_model
        self.restore_model = restore_model
        self.dataset_config = dataset_config
        self.self_supervision_config = self_supervision_config
        self.shared_device = '/gpu:0' if self.num_gpus == 1 else cpu_device
        assert(np.mod(batch_size, num_gpus) == 0)
        self.batch_size_per_gpu = int(batch_size / np.maximum(num_gpus, 1))
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)         
        
        self.checkpoint_dir = '/'.join([self.save_dir, checkpoint_dir])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 
        
        self.model_name = model_name
        if not os.path.exists('/'.join([self.checkpoint_dir, model_name])):
            os.makedirs(('/'.join([self.checkpoint_dir, self.model_name])))         
            
        self.sample_dir = '/'.join([self.save_dir, sample_dir])
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)  
        if not os.path.exists('/'.join([self.sample_dir, self.model_name])):
            os.makedirs(('/'.join([self.sample_dir, self.model_name])))    
        
        self.summary_dir = '/'.join([self.save_dir, summary_dir])
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir) 
        if not os.path.exists('/'.join([self.summary_dir, 'train'])):
            os.makedirs(('/'.join([self.summary_dir, 'train']))) 
        if not os.path.exists('/'.join([self.summary_dir, 'test'])):
            os.makedirs(('/'.join([self.summary_dir, 'test'])))             
    
                    
    def test(self, restore_model, save_dir, is_normalize_img=True):
        print('test')
        all_folders=glob.glob(self.dataset_config['img_dir']+"/abnormal-frames-maskr-112/Train/*")
        for folderpath in all_folders:
            #processed_folders=['Abnormal001_x264','Abnormal011_x264','Abnormal021_x264','Abnormal031_x264','Abnormal041_x264','Abnormal051_x264','Abnormal061_x264','Abnormal002_x264','Abnormal012_x264','Abnormal022_x264','Abnormal032_x264','Abnormal042_x264','Abnormal052_x264','Abnormal062_x264','Abnormal003_x264','Abnormal013_x264','Abnormal023_x264','Abnormal033_x264','Abnormal043_x264','Abnormal053_x264','Abnormal063_x264','Abnormal004_x264','Abnormal014_x264','Abnormal024_x264','Abnormal034_x264','Abnormal044_x264','Abnormal054_x264','Abnormal064_x264','Abnormal005_x264','Abnormal015_x264','Abnormal025_x264','Abnormal035_x264','Abnormal045_x264','Abnormal055_x264','Abnormal065_x264','Abnormal006_x264','Abnormal016_x264','Abnormal026_x264','Abnormal036_x264','Abnormal046_x264','Abnormal056_x264','Abnormal007_x264','Abnormal017_x264','Abnormal027_x264','Abnormal037_x264','Abnormal047_x264','Abnormal057_x264','Abnormal008_x264','Abnormal018_x264','Abnormal028_x264','Abnormal038_x264','Abnormal048_x264','Abnormal058_x264','Abnormal009_x264','Abnormal019_x264','Abnormal029_x264','Abnormal039_x264','Abnormal049_x264','Abnormal059_x264','Abnormal010_x264','Abnormal020_x264','Abnormal030_x264','Abnormal040_x264','Abnormal050_x264','Abnormal060_x264']
            folder_name=os.path.basename(folderpath)
            #if folder_name in processed_folders:
            #    continue
            print(folder_name)
            tf.reset_default_graph()    
            try: 
                # creating a folder named data 
                if not os.path.exists(save_dir+'/abnormal-frames-maskr-112/Train/'+folder_name): 
                    os.makedirs(save_dir+'/abnormal-frames-maskr-112/Train/'+folder_name)
                    # if not created then raise error 
            except OSError: 
                print ('Error: Creating directory of data') 
            dataset = BasicDataset(data_list_file=self.dataset_config['data_list_file']+"/abnormal-list/Train/"+folder_name+".txt", img_dir=folderpath, is_normalize_img=is_normalize_img)
            save_name_list = dataset.data_list[:, -1]
            iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
            batch_img0, batch_img1, batch_img2 = iterator.get_next()
            img_shape = tf.shape(batch_img0)
            h = img_shape[1]
            w = img_shape[2]
            
            new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
            new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)
            
            batch_img0 = tf.image.resize_images(batch_img0, [new_h, new_w], method=1, align_corners=True)
            batch_img1 = tf.image.resize_images(batch_img1, [new_h, new_w], method=1, align_corners=True)
            batch_img2 = tf.image.resize_images(batch_img2, [new_h, new_w], method=1, align_corners=True)
            
            flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=True) 
            flow_fw['full_res'] = flow_resize(flow_fw['full_res'], [h, w], method=1)
            flow_bw['full_res'] = flow_resize(flow_bw['full_res'], [h, w], method=1)
            
            flow_fw_color = flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
            flow_bw_color = flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)
            
            restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
            saver = tf.train.Saver(var_list=restore_vars)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer()) 
            sess.run(iterator.initializer) 
            saver.restore(sess, restore_model)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)           
            for i in range(dataset.data_num):
                np_flow_fw, np_flow_bw, np_flow_fw_color, np_flow_bw_color = sess.run([flow_fw['full_res'], flow_bw['full_res'], flow_fw_color, flow_bw_color])
                write_flo('%s/abnormal-frames-maskr-112/Train/%s/%s.flo' % (save_dir, folder_name, save_name_list[i]), np_flow_fw[0])
                #write_flo('%s/flow_bw_%s.flo' % (save_dir, save_name_list[i]), np_flow_bw[0])
                #print('Finish %d/%d' % (i+1, dataset.data_num))
