import numpy as np
from imgaug import augmenters as iaa
import pathlib
import imageio


# ここに元の画像をコピーしたディレクトリを作成しておく
AUG_DIR = '../input_aug/stage1_train'

# イメージファイルを作成
training_paths = pathlib.Path(AUG_DIR).glob('*/images/*.png')
training_sorted = sorted([x for x in training_paths])

# Augumentationの処理を記載
seq = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 1.0)) # blur images with a sigma of 0 to 3.0
])

# それぞれの画像に処理を加えて上書き
for img_path in training_sorted:
    # print(img_path)
    
    img = imageio.imread(str(img_path))
    img_aug = seq.augment_images(img)
    imageio.imwrite(str(img_path), img_aug)

print("Finished!")
