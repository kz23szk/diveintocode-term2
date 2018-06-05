import numpy as np
# from keras.datasets import mnist
import gc
# カラー画像のデータセット
from keras.datasets import cifar10
import dnn


(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()

# プロトタイプなので100データだけ使用する
x_train_cifar10 = x_train_cifar10[:100]
y_train_cifar10 = y_train_cifar10[:100]
y_label_cifar10 = y_train_cifar10
y_train_cifar10 = np.identity(10)[y_train_cifar10]
del x_test_cifar10, y_test_cifar10

gc.collect()


model = dnn.DNN(iteration=10, optimizer='adam', hidden_layer_list=[100], batch_norm=True)

params = {'optimizer': 'adam',
#         'batch_mode' : 'mini',
#         'init': 'he',
          'lr': 0.01}

pred_y = model.train(x_train_cifar10, y_train_cifar10.reshape(100,10), params)

print(pred_y)
