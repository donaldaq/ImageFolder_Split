#image resize

import cv2
import numpy as np
import os
import pandas as pd
import time
import operator
import collections
from datetime import datetime 

since = time.time()
now = datetime.now()


def image_path():
    dirname = "/home/mlm08/ml/data/missing_eyelid_images"
    #dirname = "/home/mlm08/ml/data/tmp"
    #dirname = "C:\\Users\\USER\\Desktop\\LSIL\\add"
    filenames = os.listdir(dirname)
    #only_name = filename.split('.')[0]
    height = []
    width = []
    duplicate = {}
    n = 0
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            try:
                extname = filename.split('.')[1]
                # if extname == 'png' or extname == 'jpg' or extname == 'jpeg' or extname == 'tif':
                if extname in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'PNG', 'JPG', 'JPEG', 'TIF', 'TIFF']:    
                    fullpathname = os.path.join(path,filename)
                    only_name = filename.split('_')[0]

                    #only_name = filename.split(' ')[0]
                    #only_name = only_name.split('(')[0]
                    #only_name = only_name.upper()
                    print(only_name)

                    #print("check filename: {}".format(fullpathname))
                    n += 1
                    try: duplicate[only_name]+= 1
                    except: duplicate[only_name]=1
            except:
                None
        # if duplicate:
        #     only_class = path.split('/')[-1]
        #     only_dataset = path.split('/')[-2]
        #     #print("each dict {}".format(only_class), duplicate)
        #     print('{}_{} duplicate count: '.format(only_dataset,only_class), str(len(duplicate)))
        #     each_duplicate = collections.OrderedDict(sorted(duplicate.items(), key=lambda x: x[1], reverse= True))
        #     df = pd.DataFrame(each_duplicate.items(), columns=["patient_id", "numbers"])
        #     df.to_csv(r'Patient_id_{}_0413.csv'.format(only_class), index= False)
    #duplicate = sorted(duplicate.items(), key=f2)
    #duplicate = sorted(duplicate.items(), key=(lambda x: x[1]), reverse= True)
    #duplicate = sorted(duplicate.items(), key=operator.itemgetter(0))
    #duplicate = collections.OrderedDict(duplicate)
    duplicate = collections.OrderedDict(sorted(duplicate.items(), key=lambda x: x[1], reverse= True))

    print('duplicate: ', duplicate)
    print('duplicate count: ', str(len(duplicate)))
    print('image count: ', n)

    month_date = str(now.month) + str(now.day)
    if len(month_date) == 3:
        month_date = '0' + str(now.month) + str(now.day)

    
    df = pd.DataFrame(duplicate.items(), columns=["patient_id", "numbers"])
    df.to_csv(r'Patient_id_{}.csv'.format(month_date), index= False)

# def image_each_path():
#     dirname = "/home/mlm08/ml/data/lid_6class/whole_dataset"
#     #dirname = "/home/mlm08/ml/data/tmp"
#     #dirname = "C:\\Users\\USER\\Desktop\\LSIL\\add"
#     each_names = os.listdir(dirname)
#     #only_name = filename.split('.')[0]

#     height = []
#     width = []
#     duplicate = {}
#     for i in range(6):
#         dup_name = i + "dup"
        

#     for (path, dir, files) in os.walk(dirname):
#         i = 0
#         merge_name = dirname + '/' + each_names[i]
#         if path == merge_name:
            
#             for filename in files:
#                 try:
#                     extname = filename.split('.')[1]
#                     # if extname == 'png' or extname == 'jpg' or extname == 'jpeg' or extname == 'tif':
#                     if extname in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'PNG', 'JPG', 'JPEG', 'TIF', 'TIFF']:    
#                         fullpathname = os.path.join(path,filename)
#                         #only_name = filename.split('_')[0]

#                         only_name = filename.split(' ')[0] #for CIN
#                         only_name = only_name.split('(')[0] #for CIN

#                         print("check filename: {}".format(fullpathname))
#                         n += 0
#                         try: duplicate[only_name]+= 1
#                         except: duplicate[only_name]=1
                        
#                 except:
#                     None
#     #duplicate = sorted(duplicate.items(), key=f2)
#     #duplicate = sorted(duplicate.items(), key=(lambda x: x[1]), reverse= True)
#     #duplicate = sorted(duplicate.items(), key=operator.itemgetter(0))
#     #duplicate = collections.OrderedDict(duplicate)
#     duplicate = collections.OrderedDict(sorted(duplicate.items(), key=lambda x: x[1], reverse= True))

#     print('duplicate: ', duplicate)
#     print('duplicate count: ', str(len(duplicate)))
#     print('image count: ', n)

    
#     df = pd.DataFrame(duplicate.items(), columns=["patient_id", "numbers"])
#     df.to_csv(r'Patient_id.csv', index= False)


if __name__ == "__main__":
    image_path()
    time_elapsed = time.time() - since
    print('time elapsed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Yey Done!!")
