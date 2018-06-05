import numpy as np
import util


class Layer:
    def __init__(self, params={}):
        if 'input_shape' in params:
            self.in_shape = params['input_shape']
        else:
            self.in_shape = None

        if 'output_shape' in params:
            self.out_shape = params['output_shape']
        else:
            self.out_shape = None

class MaxPooling(Layer):

    def __init__(self, pool_size=4, stride=-1, pad=0, params={}):
        super(MaxPooling, self).__init__(params)
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad
        self.stride = self.pool_size if self.stride == -1 else stride
        # パディング方法をしてできたほうがよい？　今はゼロ埋め固定
        # 最大値の場所を保存して、backwardでその場所以外は伝播しない
        self.max_index = None
        self.input_shape = None  # 親クラスで定義するのでここには後々必要なくなる

    def initialize(self, in_shape, params={}):
        self.in_shape = in_shape  # N H W Cで来ると想定
        N, H, W, C = in_shape
        # 出力の高さと幅を計算する
        out_h = 1 + int((H + 2 * self.pad - self.pool_size) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - self.pool_size) / self.stride)

        self.out_shape = (N, out_h, out_w, C)

        # 次の層の初期化のために出力shapeを返す
        print("Pooling : out {} filter {} stride {} pad {} ".format(self.out_shape, self.pool_size, self.stride,
                                                                    self.pad))
        return self.out_shape

    def forward(self, x):
        self.input_shape = x.shape
        N, H, W, C = x.shape

        col = util.im2col(x, self.pool_size, self.stride, self.pad)
        col_max = np.max(col, axis=1)
        self.max_index = np.argmax(col, axis=1)
        print("col {} col_max {} max_index {}".format(col, col_max, self.max_index))

        # 出力サイズを確認
        out_h = (H + 2 * self.pad - self.pool_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_size) // self.stride + 1

        # 整形
        return col_max.reshape(C, N, out_h, out_w).transpose(1, 2, 3, 0)

    def backward(self, dout):
        filter_size = dout.shape[1]
        dout_line = dout.transpose(3, 0, 1, 2).reshape(-1)

        # 返り値の箱を作る
        ret = np.zeros([dout_line.shape[0], self.pool_size * self.pool_size])
        print(ret.shape)
        print(self.max_index.shape)

        # 最大値の場所にdoutを流し込む
        for i, max_i in enumerate(self.max_index):
            ret[i, max_i] = dout_line[i]

        # 元の形状に戻してリターン
        return util.col2im(ret, self.input_shape, self.pool_size, self.stride, self.pad)


class Convolution(Layer):

    def __init__(self, out_channel=1, filter_size=3, stride=1, pad=0, bias=True, params={}):
        # biasなしに対応していない。 dbを更新しなければよい？
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.bias = bias
        self.W = None
        self.b = None
        ##### Affineからパクリ ########
        self.x = None
        # パラメータの微分値
        self.dW = None
        self.db = None
        self.optimize = None
        # 学習率のセット
        if 'lr' in params:
            self.lr = params['lr']
        else:
            self.lr = 0.01
        self.x_2dim = None  # im2col後のxの値を保持

    def initialize(self, in_shape, params={}):
        self.in_shape = in_shape  # N H W Cで来ると想定
        N, H, W, C = in_shape
        # 出力の高さと幅を計算する
        out_h = 1 + int((H + 2 * self.pad - self.filter_size) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - self.filter_size) / self.stride)

        self.out_shape = (N, out_h, out_w, self.out_channel)

        # 重みの初期化
        # im2colを見越して２次元の重みとして実装
        self.W = np.random.randn(self.filter_size * self.filter_size * self.in_shape[3], self.out_channel)
        self.b = np.zeros([self.out_channel, 1])
        # TODO 初期化を選択できるように設定する　ひとまずはガウス初期化
        #         if params['init'] == 'gauss':
        self.W *= 0.01
        #         elif params['init'] == 'xavier':
        #             # 入力層のユニット数 N * out_h * out_w ?
        #             self.W /= np.sqrt(self.out_shape[0] * self.out_shape[1] * self.out_shape[2])
        #         else: # He
        #             self.W = self.W / np.sqrt(self.out_shape[0] * self.out_shape[1] * self.out_shape[2]) * np.sqrt(2)

        # 更新式のスイッチング
        # optimizeメソッドをoptimizerによって切り替える。
        if 'optimizer' in params:
            if params['optimizer'] == 'sgd':
                self.optimize = self.update_sgd
            elif params['optimizer'] == 'adagrad':
                self.h = np.zeros_like(W)
                self.optimize = self.update_adagrad
            else:  # params['optimizer'] == 'adam':
                self.m = np.zeros_like(self.W)
                self.v = np.zeros_like(self.W)
                self.beta1 = 0.9
                self.beta2 = 0.999
                self.optimize = self.update_adam
        else:
            self.optimize = self.update_sgd

        # 次の層の初期化のために出力shapeを返す
        print(
            "Conv : out {} filter {} stride {} pad {} ".format(self.out_shape, self.filter_size, self.stride, self.pad))
        return self.out_shape

    def forward(self, x):
        in_C = self.in_shape[3]
        out_C = self.out_shape[3]
        N = self.in_shape[0]

        # 出力の高さと幅
        out_h, out_w = self.out_shape[1], self.out_shape[2]

        # ２次元配列に変換する
        # (out_h * out_w * N * C, filter_size*filter_size)
        #         print("x.shape {}".format(x.shape))
        x_2dim = util.im2col(x, self.filter_size, self.stride, self.pad)
        #         print("x_2dim.shape {}".format(x_2dim.shape))
        #         print("filter_size {} stride {} pad {}".format(self.filter_size, self.stride, self.pad))
        self.x_2dim = x_2dim

        # 各色（チャネル）ごとに重みを分割
        fil_size = self.filter_size
        W_color = self.W.reshape(in_C, fil_size * fil_size, out_C)
        # W_color = self.W.reshape(in_C, -1, out_C)
        x_2dim_color = x_2dim.reshape(in_C, fil_size * fil_size, -1)
        # x_2dim_color = x_2dim.reshape(in_C, -1, out_h * out_w * N)

        # 入力チャネルごとに対応するフィルターで計算する
        out = np.zeros((in_C, out_C, out_h * out_w * N))
        for c in range(in_C):
            out[c] = np.dot(W_color[c].T, x_2dim_color[c])

        # 入力チャネルごとの結果を合計する
        out = np.sum(out, axis=0)

        out = out + self.b

        # 整形する　out_C N H W
        return out.reshape(out_C, N, out_h, out_w).transpose(1, 2, 3, 0)

    def backward(self, dout):
        out_c = self.out_shape[3]
        dout = dout.reshape(-1, out_c)

        self.db = np.sum(dout, axis=0)[:, np.newaxis]

        # 入力チャネルの数
        in_c = self.in_shape[3]
        # sum前の状態に戻す　（入力チャネル分複製する）
        # dout_before_sum = np.tile(dout, (in_c, 1))
        # self.dW = np.dot(self.x_2dim.T, dout_before_sum)

        # f * f * out_c
        self.dW = np.dot(self.x_2dim.reshape(-1, self.filter_size * self.filter_size * in_c).T, dout)

        dout = np.dot(dout, self.W.T)

        # (R G B)と横並びなので
        #  R
        #  G
        #  B　と縦方向に変換する
        dout = np.vstack(np.split(dout, in_c, axis=1))

        dout = util.col2im(dout, self.in_shape, self.filter_size, self.stride, self.pad)
        return dout

    #####affineからパクリ
    def update_sgd(self):
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.db

    # adagrad 少しずつ更新量が減っていく
    def update_adagrad(self, lr=0.01):
        self.h += self.dW ** 2
        self.W -= self.lr * self.dW / (np.sqrt(self.h) + 1e-7)
        self.b -= self.lr * self.db

    def update_adam(self, lr=0.01):
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.dW
        self.v = self.beta2 * self.v + (1 - self.beta2) * (self.dW * self.dW)

        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)

        self.W -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        self.b -= self.lr * self.db


class Flatten(Layer):
    # (N, H, W, C)のデータを
    # (N, H * W * C)に変更する
    def __init__(self, params={}):
        super(Flatten, self).__init__(params)
        # 入出力サイズ以外のパラメータはない

    def initialize(self, in_shape, params={}):
        self.in_shape = in_shape  # N H W Cで来ると想定
        N, H, W, C = in_shape
        self.out_shape = (N, H * W * C)

        # 次の層の初期化のために出力shapeを返す
        print("Flatten : out {} ".format(self.out_shape))
        return self.out_shape

    def forward(self, x):
        self.input_shape = x.shape
        out = np.array([elem.flatten() for elem in x])
        return out

    def backward(self, dout):
        dout = dout.reshape(self.input_shape)

        return dout


class Dropout(Layer):
    def __init__(self, params):
        super(Dropout, self).__init__(params)
        if 'dropout_ratio' in params:
            self.dropout_ratio = params['dropout_ratio']
        else:
            self.dropout_ratio = 0.5
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # TODO 動作を確認する
            self.mask = np.random.rand(x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNorm(Layer):
    def __init__(self, params):
        super(BatchNorm, self).__init__(params)
        self.out = None
        self.beta = 0.0
        self.gamma = 1.0
        self.lr = params['lr']
        self.eps = 1e-8

        '''
        計算式は下記を参照
        https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        '''

    def initialize(self, in_shape, params={}):
        self.in_shape = in_shape  # N H W Cで来ると想定
        self.out_shape = self.in_shape

        # 次の層の初期化のために出力shapeを返す
        print("BatchNorm : out {} lr {}  ".format(self.out_shape, self.lr))
        return self.out_shape

    def forward(self, x):
        # data_size, input_size = x.shape

        # 単に標準化する
        # out = (x - np.mean(x, axis=0)) / np.var()

        # step1: 平均を求める
        mu = np.mean(x, axis=0)

        # step2: 偏差
        self.xmu = x - mu

        # step3 : 偏差の２乗
        sq = self.xmu ** 2

        # step4 : 分散を求める
        self.var = np.var(x, axis=0)

        # step5 : 分散のルートを取った値を求める
        self.sqrtvar = np.sqrt(self.var + self.eps)

        # step6 : sqrtvarの逆数（invert）
        self.ivar = 1.0 / self.sqrtvar

        # step7 : 標準化した値
        self.xhat = self.xmu * self.ivar

        # step8
        gammax = self.gamma * self.xhat

        # step9
        out = gammax + self.beta

        return out

    def backward(self, dout=1):
        # get the dimensions of the input/output
        N, D = dout.shape

        # step9
        self.d_beta = np.sum(dout, axis=0)
        dgammax = dout  # not necessary, but more understandable

        # step8
        self.d_gamma = np.sum(dgammax * self.xhat, axis=0)
        dxhat = dgammax * self.gamma

        # step7
        divar = np.sum(dxhat * self.xmu, axis=0)
        dxmu1 = dxhat * self.ivar

        # step6
        dsqrtvar = -1. / (self.sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(self.var + self.eps) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * self.xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2

        return dx

    def optimize(self):
        self.gamma -= self.lr * self.d_gamma
        self.beta -= self.lr * self.d_beta


class Activation(Layer):
    '''
    活性化関数を設定できる
    'tanh'
    'sigmoid'
    'relu'
    '''

    def __init__(self, params):
        super(Activation, self).__init__(params)
        self.out = None
        self.mask = None
        # optimizeメソッドを
        if 'activation' in params:
            if params['activation'] == 'tanh':
                self.forward = self.forward_tanh
                self.backward = self.backward_tanh
            elif params['activation'] == 'sigmoid':
                self.forward = self.forward_sigmoid
                self.backward = self.backward_sigmoid
            else:  # params['activation'] == 'relu':
                self.forward = self.forward_relu
                self.backward = self.backward_relu
        else:
            params['activation'] = 'relu'
            self.forward = self.forward_relu
            self.backward = self.backward_relu

    def initialize(self, in_shape, params={}):
        self.in_shape = in_shape  # N H W Cで来ると想定
        self.out_shape = self.in_shape

        # 次の層の初期化のために出力shapeを返す
        print("Activation : out {}   func : {} ".format(self.out_shape, params['activation']))
        return self.out_shape

    def forward_relu(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward_relu(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

    # tanh
    def forward_tanh(self, x):
        out = np.tanh(x)
        self.out = out

        return out

    def backward_tanh(self, dout):
        dx = dout * (1 - np.tanh(dout) ** 2)

        return dx

    # sigmoid関数
    def forward_sigmoid(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward_sigmoid(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine(Layer):

    def __init__(self, unit_size=100, params={}):
        super(Affine, self).__init__(params)
        self.W = None
        self.b = None
        self.x = None
        self.unit_size = unit_size
        # パラメータの微分値
        self.dW = None
        self.db = None
        # 学習率のセット
        if 'lr' in params:
            self.lr = params['lr']
        else:
            self.lr = 0.01

    def initialize(self, in_shape, params={}):
        self.in_shape = in_shape  # N F(特徴数)で来ると想定
        N, F = in_shape

        self.out_shape = (N, self.unit_size)

        # 重みの初期化
        # im2colを見越して２次元の重みとして実装
        self.W = np.random.randn(F, self.unit_size)
        self.b = np.zeros([1, self.unit_size])
        # TODO 初期化を選択できるように設定する　ひとまずはガウス初期化
        #         if params['init'] == 'gauss':
        self.W *= 0.01
        #         elif params['init'] == 'xavier':
        #             # 入力層のユニット数 N * out_h * out_w ?
        #             self.W /= np.sqrt(self.out_shape[0] * self.out_shape[1] * self.out_shape[2])
        #         else: # He
        #             self.W = self.W / np.sqrt(self.out_shape[0] * self.out_shape[1] * self.out_shape[2]) * np.sqrt(2)

        # オプティマイザの設定
        # optimizeメソッドをoptimizerによって切り替える。
        if 'optimizer' in params:
            if params['optimizer'] == 'sgd':
                self.optimize = self.update_sgd
            elif params['optimizer'] == 'adagrad':
                self.h = np.zeros_like(self.W)
                self.optimize = self.update_adagrad
            else:  # params['optimizer'] == 'adam':
                self.m = np.zeros_like(self.W)
                self.v = np.zeros_like(self.W)
                self.beta1 = 0.9
                self.beta2 = 0.999
                self.optimize = self.update_adam
        else:
            params['optimizer'] = 'adam'
            self.optimize = self.update_adam

        # 次の層の初期化のために出力shapeを返す
        print("Affine : out {} optimizer {} unit {} ".format(self.out_shape, params['optimizer'], self.unit_size))
        return self.out_shape

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout=1):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

    def update_sgd(self):
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.db

    # adagrad 少しずつ更新量が減っていく
    def update_adagrad(self, lr=0.01):
        self.h += self.dW ** 2
        self.W -= self.lr * self.dW / (np.sqrt(self.h) + 1e-7)
        self.b -= self.lr * self.db

    def update_adam(self, lr=0.01):
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.dW
        self.v = self.beta2 * self.v + (1 - self.beta2) * (self.dW * self.dW)

        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)

        self.W -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        self.b -= self.lr * self.db


class SoftmaxWithLoss(Layer):
    def __init__(self, params={}):
        super(SoftmaxWithLoss, self).__init__(params)
        self.loss = None  # 損失関数
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ（one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # delta3に相当

        return dx


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy_error(y, y_pred):
    data_size = y.shape[0]

    # クロスエントロピー誤差関数　y_predは０になりえるので -inf にならないためにすごく小さい補正値を入れる
    cross_entorpy = -np.sum(y * np.log(y_pred + 1e-7))

    error = cross_entorpy / data_size
    return error