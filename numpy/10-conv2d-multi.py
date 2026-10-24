'''
多通道卷积操作
'''


import numpy as np

def conv2d_multi(X, W, bias=None, stride=1, padding=0):
    '''
    X (Cin, H, W_in)
    W (Cout, Cin_w, Kh, Kw)
    '''
    Cin, H, W_in = X.shape
    Cout, Cin_w, Kh, Kw = W.shape

    # 1. 计算输出高宽
    Oh = (H + 2 * padding - Kh) // stride + 1
    Ow = (W_in + 2 * padding - Kw) // stride + 1

    # 2. Padding（仅空间维度） (0, 0) 表示通道维度不用pad
    X = np.pad(X, ((0, 0), (padding, padding), (padding, padding)), mode="constant")

    # 3. 创建输出张量
    out = np.zeros((Cout, Oh, Ow)) # (Cout, Oh, Ow)

    # 4. 遍历输出通道、输出行、输出列
    for c in range(Cout):
        for i in range(Oh):
            for j in range(Ow):
                start_h = i * stride
                start_w = j * stride
                # window
                window = X[:, start_h:start_h+Kh, start_w:start_w+Kw] # (Cin, Kh, Kw)
                value = np.sum(window * W[c, :, :, :])
                if bias is not None:
                    value += bias[c]
                out[c, i, j] = value


    return out


if __name__ == "__main__":
    X = np.ones((3, 5, 5))
    W = np.ones((8, 3, 3, 3))
    bias = np.arange(8)

    Y = conv2d_multi(X, W, bias)

    print(Y.shape)
    # (8, 3, 3)

    print(Y[:, 0, 0])
    # [27. 28. 29. 30. 31. 32. 33. 34.]