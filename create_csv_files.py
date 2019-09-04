#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:37:40 2019

@author: viswanatha
"""

from data_utils import append_new_data_to_csv, create_data_lists

create_data_lists("2019-02-19_label/vott-json-export/","2019-02-19_label/vott-json-export/panama-export.json","dataset")

append_new_data_to_csv("Data_20190618-163001_left.mp4_","Data_20190618-163001_left.mp4_/json_data/", 'dataset')
append_new_data_to_csv("Data_20190618-170001_right.mp4_","Data_20190618-170001_right.mp4_/json_data/", 'dataset')
