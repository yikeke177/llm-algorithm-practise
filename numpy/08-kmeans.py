import numpy as np

def kmeans_step(X, centroids):
    '''
    X : (N, D) 每个点
    centroids : (K, D)
    K : 聚类中心数量
    '''
    N, D = X.shape
    K = centroids.shape[0]

    # 1. 广播计算每个样本到每个中心的平方距离
    diff = X[:, None, :] - centroids[None, :, :] # (N, K, D)
    distance = np.sum(diff ** 2, axis=-1) # (N, K)

    # 2. argmin 分配聚类标签
    labels = np.argmin(distance, axis=-1)

    # 3. 更新每个聚类中心（处理空聚类）
    new_centroids = np.zeros((K, D))
    for i in range(K): 
        points = X[labels == i]
        if len(points) > 0:
            new_centroids[i] = points.mean(axis = 0)

    # 4. 返回 labels 和 new_centroids
    return labels, new_centroids


def kmeans(X, K, max_iter=100, tol=1e-4):
    '''
    X : (N, D)
    
    '''
    N, D = X.shape
    rng = np.random.default_rng(42)

    # 1. 随机初始化 K 个中心 有别的方式吗？
    indices = rng.choice(N, size=K, replace=False) # (K,)
    centers = X[indices].copy() # (K, D)

    # 2. 循环执行分配和更新
    for _ in range(max_iter):
        # 计算距离与 labels
        diff = X[:, None, :] - centers[None, :, :] # (N, K, D)
        distance = np.sum(diff ** 2, axis=-1) # (N, K)
        # 每个点最小距离的索引
        labels = np.argmin(distance, axis=-1) # (N,)
        # 更新聚类中心，处理空聚类
        new_centers = centers.copy()
        for i in range(K):
            points = X[labels == i]

            if len(points) > 0:
                new_centers[i] = points.mean(axis=0)

        centers = new_centers

        # 判断收敛
        shift = np.linalg.norm(new_centers - centers,axis=1)
        max_shift = np.max(shift)
        centers = new_centers

        if max_shift < tol:
            break
        # 方式二：
        # if np.sqrt(np.sum((new_centers - centers) ** 2)) < tol:
        #     break

    # 3. 根据最终中心重新计算 labels
    diff = X[:, None, :] - centers[None, :, :] # (N, K, D)
    distance = np.sum(diff ** 2, axis=-1) # (N, K)
    # 每个点最小距离的索引
    labels = np.argmin(distance, axis=-1) # (N,)

    # 4. 返回 labels、centroids
    return labels, centers