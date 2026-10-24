import numpy as np

def cross_entropy_loss(logits, labels):
    # 1. 获取 Batch Size
    B = logits.shape[0]

    # 2. 每行减去最大值
    logits = logits - np.max(logits, axis=-1, keepdims=True)

    # 3. 计算 LogSumExp
    log_sum_exp = np.log(
        np.sum(np.exp(logits), axis=-1)
    )

    # 4. 提取真实类别对应的 Logit
    # 错误写法 logits[:, labels] 这相当于每行都去label的那几个数
    true_logits = logits[np.arange(B), labels]

    # 5. 计算每个样本的 Loss
    loss = log_sum_exp - true_logits

    # 6. 返回平均 Loss
    return loss.mean()