#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:14:31 2019

@author: viswanatha
"""

import json
import os
import pandas as pd
import glob


custom_labels = {'cobia'}
label_map = {k: v + 1 for v, k in enumerate(custom_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()} # Inverse mapping


def coco_to_csv(images_path, annotation_path, output_folder):
    '''
    Converts coco json file data into csv files.
    csv files are useful for creating tf records.
    '''
    path = os.path.abspath(images_path) + '/'
    fp = open(annotation_path)
    json_data = json.load(fp)
    
    count = 1
    image_not_exists = 0
    images_without_boxes = 0
    images_without_boxes_image_exists = 0
    
    xml_list_train = []
    xml_list_val = []
    
    for key, value in json_data['assets'].items():
        # if these is atleast one bounding box
        if json_data['assets'][key]['regions']!=[]:
            
            width = json_data['assets'][key]['asset']['size']['width']
            height = json_data['assets'][key]['asset']['size']['height']
            # Column names for the csv file
            column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
            
            #If image exists locally
            if os.path.isfile(path + json_data['assets'][key]['asset']['name']) ==True:
               
                #for every 5 images 1 image is taken as a val image
                if count%5==0:
                    num_boxes = len(json_data['assets'][key]['regions'])
                    for box in range(num_boxes):
                        img_name = path + json_data['assets'][key]['asset']['name']
                        label = json_data['assets'][key]['regions'][box]["tags"]
                        label = label[0]
                        bou_box = json_data['assets'][key]['regions'][box]['boundingBox']
                        height  = bou_box['height']
                        width   = bou_box['width']
                        left    = bou_box['left']
                        top     = bou_box['top']
                        
                        ymin, xmin, ymax, xmax = [top, left, height, width]
                        value = (img_name,
                         xmin,
                         ymin,
                         xmax+xmin,
                         ymax+ymin,
                         label
                         )
                        xml_list_val.append(value)
                    count+=1
                    continue;
                    
                num_boxes = len(json_data['assets'][key]['regions'])
                for box in range(num_boxes):
                    img_name = path + json_data['assets'][key]['asset']['name']
                    label = json_data['assets'][key]['regions'][box]["tags"]
                    label = label[0]
                    bou_box = json_data['assets'][key]['regions'][box]['boundingBox']
                    height  = bou_box['height']
                    width   = bou_box['width']
                    left    = bou_box['left']
                    top     = bou_box['top']
                    
                    ymin, xmin, ymax, xmax = [top, left, height, width]
                    value = (img_name,
                         xmin,
                         ymin,
                         xmax+xmin,
                         ymax+ymin,
                         label
                         )
                    xml_list_train.append(value)
                    
                count += 1
            else:
                image_not_exists +=1
        else:
            images_without_boxes+=1
            if os.path.isfile(path + json_data['assets'][key]['asset']['name']) ==True:
                images_without_boxes_image_exists +=1
              
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    xml_df_train = pd.DataFrame(xml_list_train)
    xml_df_val = pd.DataFrame(xml_list_val)
    
    xml_df_train.to_csv((output_folder + '/train' + '_labels.csv'), index=None, header=None)
    xml_df_val.to_csv((output_folder + '/test' + '_labels.csv'), index=None, header=None)
    print('Successfully converted from json to csv.')
        
    print ("Train data :", xml_df_train.shape) #(Total_num_of_objects_in_train_set, 7)
    print ("Val data :", xml_df_val.shape)     #(Total_num_of_objects_in_val_set, 7)

def create_data_lists(images_path, annotation_path, output_folder):
    """
    :param images_path: path to the images folder
    :param annotation_path: path to the json file
    :param output_folder: folder where the csv files must be saved
    """
    coco_to_csv(images_path, annotation_path, output_folder)

def append_new_data_to_csv(images_path, annotation_path, output_folder):
  train_df = pd.read_csv(output_folder + '/train_labels.csv', header=None)
  test_df  = pd.read_csv(output_folder + '/test_labels.csv', header=None)

  print ("Train data::", train_df.shape, "Test data::", test_df.shape)

  train_df_len = len(train_df)
  test_df_len  = len(test_df)

  count = 1
  img_files = glob.glob(images_path+"/*.jpg")
  
  train_objects = 0
  train_images_count = 0
  val_objects = 0
  val_images_count = 0
  
  for img_file in img_files:
    
    img_file_split = img_file.split("/")
    img_name = img_file_split[-1]
    img_name_split = img_name.split(".")[0]
    json_file = annotation_path + img_name_split + ".json"
    
    
    try:
      json_data = json.load(open(json_file))
    except FileNotFoundError as e:
      continue;
      
    # For every 5 images one image is val image
    if count%5 == 0:
      num_boxes = len(json_data['shapes'])
      val_objects += num_boxes
      val_images_count +=1
      for box_index in range(num_boxes):
        xmin = json_data['shapes'][box_index]['points'][0][0]
        ymin = json_data['shapes'][box_index]['points'][0][1]
        xmax = json_data['shapes'][box_index]['points'][1][0]
        ymax = json_data['shapes'][box_index]['points'][1][1]
        label = json_data['shapes'][box_index]['label']
        value = (img_file,
                xmin,
                ymin,
                xmax,
                ymax,
                label
                )
        test_df.loc[test_df_len+1] = value
        test_df_len +=1
        #print (count%5, count, value)
      count+=1
    else:
      num_boxes = len(json_data['shapes'])
      train_objects += num_boxes
      train_images_count += 1
      for box_index in range(num_boxes):
        xmin = json_data['shapes'][box_index]['points'][0][0]
        ymin = json_data['shapes'][box_index]['points'][0][1]
        xmax = json_data['shapes'][box_index]['points'][1][0]
        ymax = json_data['shapes'][box_index]['points'][1][1]
        label = json_data['shapes'][box_index]['label']
        value = (img_file,
                xmin,
                ymin,
                xmax,
                ymax,
                label
                )
        train_df.loc[train_df_len+1] = value
        train_df_len +=1
        #print (count%5, count, value)
      count+=1

  print (count)
  print('\nThere are %d training images containing a total of %d objects.' % (
        train_images_count, train_objects))
  
  print('\nThere are %d validation images containing a total of %d objects.' % (
        val_images_count, val_objects))
  
  print ("Train data::", train_df.shape)
  print ("Test data::", test_df.shape)
        #print (img_file, xmin)
    #print (img_file, json_file)
    
  train_df.to_csv((output_folder + '/train' + '_labels.csv'), index=None, header=None)
  test_df.to_csv((output_folder + '/test' + '_labels.csv'), index=None, header=None)
  print('Successfully converted from json to csv.')