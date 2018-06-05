import numpy as np


def im2col(x, filter_size=4, stride=4, pad=0, padding='constant'):
    # 4次元データを2次元データに変形する
    # XXX: もしかしたらできているかも？？　今の所パディングして数が変わるとエラーになる
    # TODO: 上記を修正して、畳み込み層にも使えるように修正する

    # パディングして外堀を埋める
    if pad > 0:
        x = np.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)], padding)

    # データ数　高さ　幅　チャネルを取得
    N, H, W, C = x.shape

    # それぞれのブロックごとにflatten
    return np.array([x[n, h:h + filter_size, w:w + filter_size, c].flatten() for c in range(C) for n in range(N) \
                     for h in range(0, H - filter_size + 1, stride) for w in range(0, W - filter_size + 1, stride)])


def im2col_v2(x, filter_size=4, stride=4, pad=0, padding='constant'):
    # 4次元データを2次元データに変形する
    # 畳み込み層用の正式なim2col

    # パディングして外堀を埋める
    if pad > 0:
        x = np.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)], padding)

    # データ数　高さ　幅　チャネルを取得
    data_size, height, width, channel = x.shape

    # それぞれのブロックごとにflatten
    return np.array([x[n, h:h + filter_size, w:w + filter_size, :].flatten() for n in range(data_size) \
                     for h in range(0, height - filter_size + 1, stride) for w in range(0, width - filter_size + 1, stride)])


def col2im_v2(x, input_shape, filter_size=4, stride=4, pad=0, padding='constant'):
    # im2colの逆をやる関数
    # 畳み込み層用の正式なim2col
    # 戻すデータの形状

    _, out_H, out_W, _ = x.shape
    in_N, in_H, in_W, in_C = input_shape

    # リターン箱を作る
    img = np.zeros((in_N, in_H + 2*pad, in_W + 2*pad, in_C))

    for i, line in enumerate(x):
        data_i, square_i = divmod(i, out_H*out_W)
        height_i, width_i = divmod(square_i, out_H)
        img[data_i, height_i + filter_size, width_i + filter_size, :] = line.reshape(filter_size, filter_size, in_C)

    # paddingを削る
    return img[:, pad:pad + in_H, pad:pad + in_W, :]


# im2colの逆をやる関数
def col2im(x, input_shape, filter_size=4, stride=4, pad=0, padding='constant'):
    # 戻すデータの形状
    N, H, W, C = input_shape

    # リターン箱を作る
    img = np.zeros((N, H + 2*pad, W + 2*pad, C))
    # img = np.zeros(input_shape)
    # ブロック数
    block_num = (pad * 2 + W - filter_size) // stride + 1  # 本当は高さ　幅別々に計算する

    # １チャネルに何行あるか
    c_vol = int(x.shape[0] / C)
    # c_vol = int(ret.shape[0] / C)
    n_vol = int(c_vol / N)

    for i, line in enumerate(x):
        # 帯からブロックにする
        block = line.reshape(filter_size, filter_size)

        channel_i = i // c_vol

        data_i = i % c_vol // n_vol

        start_h, start_w = divmod(i % n_vol, block_num)  # 本当はblock_h
        start_h, start_w = int(start_h), int(start_w)

        end_h = start_h + filter_size  # filter_size or filter_height
        end_w = start_w + filter_size  # filter_size or filter_weight

        try:
            img[data_i, start_h:end_h, start_w:end_w, channel_i] = block
        except:
            print("N {} H {} W {} C{} ".format(data_i, start_h, start_w, channel_i))
            print(block.shape)
            print("img shape {}".format(img.shape))
            print(img[data_i, start_h:end_h, start_w:end_w, channel_i].shape)
            print('i {} c_vol {} n_vol {} block_num {}'.format(i, c_vol, n_vol, block_num))
            raise

    return img[:, pad:pad + H, pad:pad + W, :]