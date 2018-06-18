#!/bin/bash

echo "START: get datasets"
cd
pip install --upgrade pip
pip install kaggle
mkdir .kaggle
echo '{"username":"kz23szk","key":"31dd49a7d6cfd0443b4e6e9a11050af4"}' > .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json
apt install git
git clone https://github.com/matterport/Mask_RCNN 

echo "Mask RCNN install"
mkdir Mask_RCNN/datasets
mkdir Mask_RCNN/datasets/nucleus
cd Mask_RCNN/datasets/nucleus 
kaggle competitions download -c data-science-bowl-2018 -w

mkdir stage1_test
mkdir stage1_train
unzip stage1_test.zip -d stage1_test
unzip stage1_train.zip -d stage1_train


echo "GPU setting"
# GPU設定
source activate tensorflow_p36
conda update -n base conda --yes
conda install -c conda-forge opencv --yes
pip install imgaug --yes
conda update numpy --yes
conda install -c anaconda scikit-image --yes

# pyファイルのある場所で以下を実行
# python3 nucleus.py train --dataset=../../datasets/nucleus/ --subset=train --weights=imagenet

echo "finish!"