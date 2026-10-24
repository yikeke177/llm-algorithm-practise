

import numpy as np

'''
实现softmax
'''
def softmax(x, axis = -1):
    '''
    计算公式
    x :(..., m, n, c)
    沿着最后一个轴进行softmax
    先减去最大值
    然后分别求分子分母
    '''
    # 先减去最大值 防止指数溢出
    x = x - np.max(x, axis=axis, keepdims=True)
    # 分子 指数运算 
    x_exp = np.exp(x)

    # 分母 指数求和 沿最后轴求和
    x_exp_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_exp_sum
    
    


