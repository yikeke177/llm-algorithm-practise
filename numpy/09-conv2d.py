'''
单通道卷积操作
'''

import numpy as np

def conv2d(X, kernel, stride=1, padding=0):
    H, W = X.shape
    Kh, Kw = kernel.shape

    # 1. 计算输出高宽
    Oh = (H + 2 * padding - Kh) // stride + 1
    Ow = (W + 2 * padding - Kw) // stride + 1

    # 2. 对输入进行零填充 两个元组分别表示上下 和 左右 分别填充多少个元素
    X = np.pad(X, ((padding, padding), (padding, padding)), mode="constant") # constant什么意思？

    # 3. 创建输出数组
    out = np.zeros((Oh, Ow))

    # 4. 遍历输出位置，提取窗口并计算卷积
    for i in range(Oh):
        for j in range(Ow):
            start_h = i * stride
            strat_w = j * stride
            # 窗口
            window = X[start_h:start_h + Kh, strat_w:strat_w+Kw]
            out[i, j] = np.sum(window * kernel)

    return out


if __name__ == '__main__':
    assert conv2d(
        np.arange(9).reshape(3, 3),
        np.ones((2, 2))
    ).shape == (2, 2)

    assert conv2d(
        np.ones((5, 5)),
        np.ones((3, 3)),
        stride=2
    ).shape == (2, 2)

    assert conv2d(
        np.ones((3, 3)),
        np.ones((3, 3)),
        padding=1
    ).shape == (3, 3)