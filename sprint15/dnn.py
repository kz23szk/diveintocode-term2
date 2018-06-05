import numpy as np
from sklearn.model_selection import train_test_split
import lenet


class DNN:
    def __init__(self, init='gauss', iteration=500, lr=0.05, lam=0.01,
                 batch_mode='mini', activation='relu',
                 batch_size_rate=0.1, hidden_layer_list=[5], optimizer='sgd',
                 batch_norm=False, dropout_ratio=0.0):
        """ ハイパーパラメータ解説
        init: 初期化方法
            'he' :
            'gauss'
            'xavier'
        lr : 学習率
        lam : 正則化項の率
        batch_size: バッチサイズ
            'batch' : フルサイズ
            'mini' 0< x< 1: フルサイズ割合 0.1なら全体の0.1サイズ使用する
            'online' : オンライン学習　１データのみ
        hidden_layer_list : 隠れ層のリスト、層のユニットをリストで入力　例[2, 3]　ユニット数２、ユニット数３の隠れ層
        optimizer : 勾配の更新手法
            'sgd' : 確率的勾配降下法
            'adam':
            'adagrad':
        activation: 活性化関数の名前
            'relu' : ReLU関数
            'tanh' : tanh
            'sigmoid' : シグモイド関数
        """
        self.params = {}
        self.params['iteration'] = iteration
        self.params['init'] = init
        self.params['lr'] = lr
        self.params['lam'] = lam  # 正則化項用の係数　今は使っていない
        self.params['batch_mode'] = batch_mode  # データ数が決まったらそれに基づいて変更する
        self.params['batch_size_rate'] = batch_size_rate  # ミニバッチ法のときのみ使用する
        self.params['hidden_layer_list'] = hidden_layer_list
        self.params['optimizer'] = optimizer
        self.params['batch_norm'] = batch_norm
        self.params['dropout_ratio'] = dropout_ratio
        self.params['activation'] = activation  # 活性化関数

    def train(self, X, y, params={}):
        # 入力パラメータがあれば更新する
        for key in params:
            self.params[key] = params[key]

        # 正規化　必要？
        X = X / 255.0

        # 訓練とテストデータに分割
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=0)

        self.params['data_size'] = X_train.shape[0]
        self.params['input_size'] = X_train.shape[1]
        self.params['output_size'] = y_train.shape[1]

        # コストや正答率の学習曲線を引くためのリストを用意
        past_train_costs = []
        past_test_costs = []
        past_train_accuracy = []
        past_test_accuracy = []

        N, H, W, C = X_train.shape

        # バッチサイズの設定
        if self.params['batch_mode'] == 'batch':
            self.params['batch_size'] = self.params['data_size']
        elif self.params['batch_mode'] == 'mini':
            self.params['batch_size'] = int(self.params['data_size'] * self.params['batch_size_rate'])
        else:
            self.params['batch_size'] = 1
        # 隠れ層やレイヤーインスタンス生成
        # 入力サイズをバッチ分に調整
        in_shape = (self.params['batch_size'], H, W, C)
        self.params['layer'] = lenet.LeNetLayers(self.params)
        # 入出力サイズ
        self.params['layer'].initialize(X_train[:self.params['batch_size']], y_train[:self.params['batch_size']],
                                        self.params)

        # 確認のためここでブレークする
        #         if True:
        #             return self.params['layer'].predict(X_train)

        # 何イテレーションで1エポックか
        epoch_per_i = int(self.params['data_size'] / self.params['batch_size'])

        ##################
        # 最急降下法での学習
        ##################
        for i in range(self.params['iteration']):

            # 学習に使用するデータをサンプリング
            choice_index = np.random.choice(self.params['data_size'], self.params['batch_size'])
            X_batch, y_batch = X_train[choice_index], y_train[choice_index]

            # 誤差逆伝播法によって勾配を求め、値を更新
            self.params['layer'].optimize(X_batch, y_batch)

            # 1エポックごとに正答率とコストを算出して保存する
            if i % epoch_per_i == 0:
                past_train_accuracy.append(self.params['layer'].accuracy(X_train, y_train))
                past_test_accuracy.append(self.params['layer'].accuracy(X_test, y_test))

                past_train_costs.append(self.params['layer'].loss(X_train, y_train))
                past_test_costs.append(self.params['layer'].loss(X_test, y_test))

        return past_train_accuracy, past_test_accuracy, past_train_costs, past_test_costs

        # 現在のパラメータで予測値を確率かラベルで出力する。

    def predict(self, X, probability=False):
        predict = self.params['layer'].predict(X, train_flg=False)
        predict_proba = softmax(predict)
        if probability == True:
            return predict_proba
        else:
            return np.argmax(predict_proba, axis=1)