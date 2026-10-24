import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)

    x_exp = np.exp(x)
    dem = np.sum(x_exp, axis=axis, keepdims=True)

    return x_exp / dem


def attention(Q, K, V):
    # 1. 获取 d_k
    d_k = Q.shape[-1]

    # 2. 计算注意力分数
    num = Q @ np.swapaxes(K, -1, -2)

    # 3. 缩放
    score = num / np.sqrt(d_k)

    # 4. 稳定版 Softmax
    weight = softmax(score, axis=-1)

    # 5. 加权聚合 V
    return weight @ V