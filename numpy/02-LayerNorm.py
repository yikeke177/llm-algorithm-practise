import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x: (batch, token, c)
    gamma、beta: (c,)
    """

    # 1. 均值
    mean = np.mean(x, axis=-1, keepdims=True)
    # 2. 方差
    var = np.var(x, axis=-1, keepdims=True)
    # 3. 分子
    num = x - mean
    # 4. 分母
    dem = np.sqrt(var + eps)
    # 5. 标准化 + 缩放平移
    return gamma * (num / dem) + beta
  
