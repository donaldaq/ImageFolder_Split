# ImageFolder_Split
Train/Validation/Test split(Group or Just)

This repository has two split codes 



- automated_group_split.py
  - Train/Validation/Test = 8:1:1
  - random seed numbers: 3 [ex) 8, 88, 888]
  - split **parsing word '_'**  [ex) groupid_imagename.jpg]
  - You can use for pytorch imagefolder dataset
- folder_split.py
  - Train/Validation/Test = 8:1:1
  - random seed numbers: 3 [ex) 8, 88, 888]
  - You can use for pytorch imagefolder dataset





### Group split folder structure

#### Folder Structure 

```buildoutcfg
+-- root
|   +-- train
|       +-- class1
|           +-- class1_img1.jpg
|           +-- class1_img2.jpg
|           +-- class1_img3.jpg
|       +-- class2
|           +-- class2_img2.jpg
|           +-- class2_img3.jpg
|       +-- class3
|   +-- test
|       +-- class1
|           +-- class1_img5.jpg
|           +-- class1_img6.jpg
|       +-- class2
|       +-- class3
|   +-- val
|       +-- class1
|       +-- class2
|           +-- class2_img4.jpg
|           +-- class2_img5.jpg
|       +-- class3
```



### Group split folder structure

#### folder structure

```buildoutcfg
+-- root
|   +-- train
|       +-- class1
|           +-- img1.jpg
|           +-- img2.jpg
|           +-- img3.jpg
|       +-- class2
|       +-- class3
|   +-- test
|       +-- class1
|       +-- class2
|       +-- class3
|   +-- val
|       +-- class1
|       +-- class2
|       +-- class3
```