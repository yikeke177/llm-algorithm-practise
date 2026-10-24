import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)

    x_exp = np.exp(x)
    dem = np.sum(x_exp, axis=axis, keepdims=True)

    return x_exp / dem

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    '''
    X (B, L, D) D = num_heads * d
    1. 
    '''
    # 1. 获取 B、L、D 和 head_dim
    B, L, D = X.shape
    head_dims = D // num_heads


    # 2. 投影得到 Q、K、V
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # 3. reshape + transpose 拆分 Head
    # (B, L, D) --> (B, L, nums_heads, d) --> (B, nums_heads, L, d)
    Q = Q.reshape(B, L, num_heads, head_dims)
    K = K.reshape(B, L, num_heads, head_dims)
    V = V.reshape(B, L, num_heads, head_dims)

    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # 4. 计算缩放点积 Attention
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(head_dims)
    weights = softmax(scores)

    out = weights @ V # (B, nums_heads, L, d)


    # 5. 合并多个 Head
    out = np.transpose(out, (0, 2, 1, 3))
    out = np.reshape(out, (B, L, D))

    # 6. 输出投影
    return out @ Wo