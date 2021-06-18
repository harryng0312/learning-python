import numpy as np

# 1.6. basic array operations
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
print(f"sum: {data+ones}")

data = np.array([[1, 2, 3, 2], [2, 4, 5, 6], [2, 1, 2, 3], [2, 1, 2, 3]])
data2= np.array([[1], [2], [3], [2]])
data += data2
print(f"sum2: {data}")
