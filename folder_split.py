'''
automated group split
'''

import os
import sys
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import random
from tqdm import tqdm
import time


## for configuration 
import yaml

ymlfile = sys.argv[1]

config_path = 'config'
yml_path = os.path.join(config_path, ymlfile)

with open(yml_path, 'r') as stream:
    cfg = yaml.safe_load(stream)


# groups_test = []
groups_np = []
groups_test_random = []
groups_np_random = []
fulls_np = []
fulls_random = []
fulls_np_random = []

half_fulls_sample = []
half_fulls_sample_np = []
#X_test = []
#y_test = []

te_test = []
va_test = []

X_test1 = []
y_test1 = []
half_sample = []

duplicate_count = {}


def image_path_split(dirname):

    #dirname = "C:\\Users\\USER\\Desktop\\LSIL\\add"
    filenames = os.listdir(dirname)

    train, xte, yt, yte = train_test_split(filenames, filenames, test_size=0.2, random_state=42)
    val, test, yt, yte = train_test_split(xte, xte, test_size=0.5, random_state=42)
    train_second, val_second, yt, yte = train_test_split(train, train, test_size=0.111, random_state=44)
    train_third, val_third, yt, yte = train_test_split(train, train, test_size=0.111, random_state=46)

    shuffle_test = []
    fulls = []
    for filename in filenames:
        fullpathname = os.path.join(dirname,filename)


        shuffle_test.append(filename)
        fulls.append(fullpathname)
        #img = cv2.imread(fullpathname)

        try: duplicate_count[filename]+= 1
        except: duplicate_count[filename]=1
    print("duplicate_count: "+str(len(duplicate_count)))

    # print(len(shuffle_test))
    # groups_np = np.array(shuffle_test)
    # fulls_np = np.array(fulls)
    # print(groups_np.shape)
    # print(len(groups_np))
    # #print(os.path.basename(fulls[1]).split(' ')[0])
    # X_test = np.array(groups_np).reshape(len(groups_np),1)
    # y_test = np.array(groups_np).reshape(len(groups_np),1)
    # print(X_test.shape)
    # print(y_test.shape)
    
    return shuffle_test, fulls

#X_test, y_test, groups_np = image_path_split() # have to change point 

def img_copy(fullfilename, filename, path_name):

    #height, width, channel = img.shape
    
    #print(filename)

    
    if not(os.path.isdir(path_name)):
        os.makedirs(os.path.join(path_name))
        
    shutil.copy(fullfilename, path_name)

def dataset_group_split(X_test, y_test, groups_np):
    
    for train_idx1, test_idx1 in gss.split(X_test, y_test, groups_np):
        
        # make array train, test for group
        new_x = X_test[train_idx1]
        new_y = y_test[test_idx1]
        tr_test = np.array(X_test).reshape(len(X_test),1)
        new_tr = tr_test[train_idx1]
    
    print(len(new_x))
    print(len(new_y))

    for train_idx2, val_idx2 in val_gss.split(new_y, new_y, new_y):
        
        # make array train, test for group
        te_test = np.array(new_y).reshape(len(new_y),1)
        va_test = np.array(new_y).reshape(len(new_y),1)
        new_t = te_test[train_idx2]
        new_v = va_test[val_idx2]

    ## test set numbers
    print('Test Length:',len(new_t))
    ## test set numbers
    print('Validation Length:',len(new_v))

    tr_va = np.concatenate((new_tr,new_v), axis=None)

    for train_idx3, val_idx3 in trainval_gss.split(tr_va, tr_va, tr_va):
        
        # make array train, test for group
        retr_test = np.array(tr_va).reshape(len(tr_va),1)
        reva_test = np.array(tr_va).reshape(len(tr_va),1)
        re_tr = retr_test[train_idx3]
        re_va = reva_test[val_idx3]

    for train_idx4, val_idx4 in trainval_gss_second.split(tr_va, tr_va, tr_va):
        
        # make array train, test for group
        retr_test = np.array(tr_va).reshape(len(tr_va),1)
        reva_test = np.array(tr_va).reshape(len(tr_va),1)
        re_tr_second = retr_test[train_idx4]
        re_va_second = reva_test[val_idx4]

    return new_x, new_t, new_v, re_tr, re_va, re_tr_second, re_va_second

def excute_group_dataset(image_dir, new_x, new_t, new_v, fulls, seednumber):
    
    test_since = time.time()
    # ../../rootdir/labelfolder
    image_dir, filename =  os.path.split(image_dir)
    #extname = filename.split('.')[1]
    
    upperfolder = os.path.dirname(image_dir)
    classfolder = filename
    seednumber = str(seednumber)
    test_path_name = upperfolder + '/' + seednumber + '/test/' + classfolder 
    val_path_name = upperfolder + '/' + seednumber +  '/val/' + classfolder 
    train_path_name = upperfolder + '/' + seednumber +  '/train/' + classfolder
        
    #len_fulls = tqdm(range(len(fulls)))
    print("test dataset processing...")                          
    for i in range(len(fulls)):
        #print("test Train idx:", os.path.basename(fulls[i]).split(' ')[0])
        train_idx = os.path.basename(fulls[i])#.split(parsing_word)[0]
        
        train_filename = os.path.basename(fulls[i])
        
        for j in range(len(new_t)):
            if new_t[j] == train_idx:
                img_copy(fulls[i],train_filename, test_path_name)

    test_time_elapsed = time.time() - test_since
    print('Complete in {:.0f}m {:.0f}s'.format(
        test_time_elapsed // 60, test_time_elapsed % 60))
    print("test dataset done!!")

    # validation group copy
    val_since = time.time()
    
    #len_fulls = tqdm(range(len(fulls)))
    
    print("validation dataset processing...")                          
    
    for i in range(len(fulls)):
        #print("test Train idx:", os.path.basename(fulls[i]).split(' ')[0])
        train_idx = os.path.basename(fulls[i])#.split(parsing_word)[0]

        train_filename = os.path.basename(fulls[i])
        
        for j in range(len(new_v)):
            if new_v[j] == train_idx:
                #img = cv2.imread(fulls[i])
                img_copy(fulls[i],train_filename, val_path_name)

    val_time_elapsed = time.time() - val_since
    print('Complete in {:.0f}m {:.0f}s'.format(
        val_time_elapsed // 60, val_time_elapsed % 60))
    print("validation dataset done!!")

    # train group copy
    train_since = time.time()
    print("Train dataset processing...")
    for i in range(len(fulls)):
        #print("test Train idx:", os.path.basename(fulls[i]).split('_')[0])
        train_idx = os.path.basename(fulls[i])#.split(parsing_word)[0]
        
        train_filename = os.path.basename(fulls[i])
        
        for j in range(len(new_x)):
            if new_x[j] == train_idx:
                #img = cv2.imread(fulls[i])
                img_copy(fulls[i],train_filename, train_path_name)
                
    train_time_elapsed = time.time() - train_since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        train_time_elapsed // 60, train_time_elapsed % 60))
    print("Train dataset done!!")

def excute_trainval_dataset(image_dir, new_x, new_t, new_v, fulls, seednumber):
    
    test_since = time.time()
    # ../../rootdir/labelfolder
    image_dir, filename =  os.path.split(image_dir)
    #extname = filename.split('.')[1]
    
    upperfolder = os.path.dirname(image_dir)
    classfolder = filename
    seednumber = str(seednumber)
    test_path_name = upperfolder + '/' + seednumber + '/test/' + classfolder 
    val_path_name = upperfolder + '/' + seednumber + '/val/' + classfolder 
    train_path_name = upperfolder + '/' + seednumber + '/train/' + classfolder

    print("Test dataset processing...")                          
    for i in range(len(fulls)):
        #print("test Train idx:", os.path.basename(fulls[i]).split(' ')[0])
        train_idx = os.path.basename(fulls[i])#.split(parsing_word)[0]
        train_filename = os.path.basename(fulls[i])
        
        for j in range(len(new_t)):
            if new_t[j] == train_idx:
                #img = cv2.imread(fulls[i])
                img_copy(fulls[i],train_filename, test_path_name)

    test_time_elapsed = time.time() - test_since
    print('Complete in {:.0f}m {:.0f}s'.format(
        test_time_elapsed // 60, test_time_elapsed % 60))
    print("Test dataset done!!")

    # validation group copy
    val_since = time.time()
    #len_fulls = tqdm(range(len(fulls)))

    print("Re-validation dataset processing...")                          
    for i in range(len(fulls)):
        train_idx = os.path.basename(fulls[i])#.split(parsing_word)[0]
        train_filename = os.path.basename(fulls[i])
        
        for j in range(len(new_v)):
            if new_v[j] == train_idx:
                #img = cv2.imread(fulls[i])
                img_copy(fulls[i],train_filename, val_path_name)

    val_time_elapsed = time.time() - val_since
    print('Complete in {:.0f}m {:.0f}s'.format(
        val_time_elapsed // 60, val_time_elapsed % 60))
    print("Validation dataset done!!")

    # Train Group Copy
    train_since = time.time()
    print("Re-train dataset processing...")
    for i in range(len(fulls)):
        
        train_idx = os.path.basename(fulls[i]) #.split(parsing_word)[0]
        train_filename = os.path.basename(fulls[i])        
        for j in range(len(new_x)):
            if new_x[j] == train_idx:
                #img = cv2.imread(fulls[i])
                img_copy(fulls[i],train_filename, train_path_name)
                
    train_time_elapsed = time.time() - train_since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        train_time_elapsed // 60, train_time_elapsed % 60))
    print("Train dataset done!!")





if __name__ == "__main__":
    
    dirname = cfg['folder_split']['dir_path']

    if cfg['folder_split']['testset_fixed']  == True:
        trainval = []
        y_test = []
        seed_number = cfg['folder_split']['seed']['first']
        trainval_seed_first = cfg['folder_split']['seed']['second']
        trainval_seed_second = cfg['folder_split']['seed']['third']
        
        gss = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed_number)
        gss.get_n_splits()

        val_gss = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed_number)
        val_gss.get_n_splits()

        trainval_gss = GroupShuffleSplit(n_splits=1, test_size=.111, random_state=trainval_seed_first)
        trainval_gss.get_n_splits()

        trainval_gss_second = GroupShuffleSplit(n_splits=1, test_size=.111, random_state=trainval_seed_second)
        trainval_gss_second.get_n_splits()

        for (path, dir, files) in os.walk(dirname):
            #print('check: ',dir)
            for i in dir:
                print('check:', i) 
                image_dir = dirname + i
                X_test, fulls= image_path_split(image_dir)
                
                train, xte, yt, yte = train_test_split(X_test, X_test, test_size=0.2, random_state=seed_number)
                val, test, yt, yte = train_test_split(xte, xte, test_size=0.5, random_state=seed_number)

                print(len(train))
                print(len(val))

                trainval.extend(train)
                trainval.extend(val)

                print(len(trainval))
            
                
                train_second, val_second, yt, yte = train_test_split(trainval, trainval, test_size=0.1111, random_state=trainval_seed_first)
                train_third, val_third, yt, yte = train_test_split(trainval, trainval, test_size=0.1111, random_state=trainval_seed_second)
                
                #new_x, new_t, new_v, re_tr, re_va, re_tr_second, re_va_second = dataset_group_split(X_test, y_test, groups_np)
                
                ####################### have to use
                excute_group_dataset(image_dir, train, test, val, fulls, seed_number)
                excute_group_dataset(image_dir, train_second, test, val_second, fulls, trainval_seed_first)
                excute_group_dataset(image_dir, train_third, test, val_third, fulls, trainval_seed_second)
                ######################## end of execution function ########################
                
    else:
        trainval = []
        y_test = []
        seed_list = []
        seed_first = cfg['folder_split']['seed']['first']
        seed_second = cfg['folder_split']['seed']['second']
        seed_third = cfg['folder_split']['seed']['third']
        #parsing_word = ' '
        #classname = '2'

        seed_list.append(seed_first)
        seed_list.append(seed_second)
        seed_list.append(seed_third)

        for index in range(len(seed_list)):

            seed_number = seed_list[index]


            gss = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed_number)
            gss.get_n_splits()

            val_gss = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed_number)
            val_gss.get_n_splits()

            for (path, dir, files) in os.walk(dirname):
                #print('check: ',dir)
                for i in dir:
                    print('Check:', i) 
                    image_dir = dirname + i
                    X_test, fulls= image_path_split(image_dir)
                    
                    train, xte, yt, yte = train_test_split(X_test, X_test, test_size=0.2, random_state=seed_number)
                    val, test, yt, yte = train_test_split(xte, xte, test_size=0.5, random_state=seed_number)

                    print('Train Length:',len(train))
                    print('Validation Length:',len(val))

                    
                                       
                    
                    ####################### have to use #######################
                    excute_group_dataset(image_dir, train, test, val, fulls, seed_number)
                    


