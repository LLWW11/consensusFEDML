import numpy as np

# 生成 0-2 内的 10 个随机整数，且每个整数的个数相等
random_ints = np.random.choice(3, 3, replace=False)
result2 = np.random.choice(3, 10 % 3, replace=False)
result1 = np.repeat(random_ints, 10//3)
result = np.concatenate([result1, result2], axis=0)


# print(result)
