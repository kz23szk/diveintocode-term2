from collections import OrderedDict
import numpy as np
import layer


class LeNetLayers:
    def __init__(self, params):
        unit_size_list = [params['input_size']]
        unit_size_list.extend(params['hidden_layer_list'])
        unit_size_list.append(params['output_size'])

        self.params = {}

        # レイヤの生成
        self.layers = OrderedDict()

        # とりあえずここにべた書きで書けるようにする
        self.layers['Conv1'] = layer.Convolution(6, 5, 1, 2)  # 2 (3,3) (1,1) same
        # self.layers['BatchNorm1'] = BatchNorm(params)
        self.layers['Active1'] = layer.Activation(params)  # relu
        self.layers['Pool1'] = layer.MaxPooling(2, 2, 0, params)

        self.layers['Conv2'] = layer.Convolution(16, 5, 1, 2)  # 2 (3,3) (1,1) same
        # self.layers['BatchNorm2'] = BatchNorm(params)
        self.layers['Active2'] = layer.Activation(params)  # relu
        self.layers['Pool2'] = layer.MaxPooling(2, 2, 0, params)

        self.layers['Flatten'] = layer.Flatten(params)

        # アフィン変換層（Wx + b）を追加する
        self.layers['Affine1'] = layer.Affine(120, params)
        self.layers['Active3'] = layer.Activation(params)  # relu
        self.layers['Affine2'] = layer.Affine(84, params)
        self.layers['Active4'] = layer.Activation(params)  # relu
        self.layers['Affine3'] = layer.Affine(params['output_size'], params)

    def initialize(self, x, y, params):
        in_shape = x.shape  # N, H, W, C
        print(x.shape)
        out_shape = y.shape  # N, Class
        print('###########' * 3)
        for i, layer_ in enumerate(self.layers.values()):
            print(" Layer {}".format(i))
            in_shape = layer_.initialize(in_shape, params)
            print('###########' * 3)

        self.lastLayer = layer.SoftmaxWithLoss()

        # self.params['hidden_layer_num'] = len(unit_size_list)-1

    def predict(self, x):
        # forwardを繰り返す
        # ソフトマックスを通さなくても答えは出るのでこれで予測とする
        #         # argmaxでラベルを取れる
        #         for layer in self.layers.values():
        #             x =layer.forward(x)
        for key, layer in self.layers.items():
            print(key)
            x = layer.forward(x)

        return x

    def accuracy(self, x, t):
        # 正答率を小数点第二桁で出力する
        y_pred = self.predict(x)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(t, axis=1)
        data_size = x.shape[0]

        correct_count = np.sum([y_true == y_pred])
        score = correct_count / data_size * 100

        return round(score, 2)

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def optimize(self, x, t):

        # forward
        self.loss(x, t)

        # backward
        dout = self.lastLayer.backward(1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # optimizeメソッドがある層は更新を行う
        # AffineとBatchNorm層のみ行うはず
        for layer in self.layers.values():
            if hasattr(layer, "optimize"):
                layer.optimize()
