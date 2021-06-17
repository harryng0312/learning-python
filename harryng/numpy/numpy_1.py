import numpy as np

# 1. array
# 1.1. initiation
a = np.array([1, 2, 3])
print(a)

a = np.zeros(2)
print(a)

a = np.ones(2)
print(a)

a = np.empty(2)
print(f"empty ", a)

# a = np.arange(4)
a = np.arange(2, 9, 2)
print(a)

a = np.linspace(0, 10, 5)
print(a)

# 1.2. adding/removing/sorting
arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
a = np.sort(arr)
print(a)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.concatenate((a, b), axis=0))
print(np.concatenate((a, b), axis=1))

# 1.3. shape
arr = np.array([
    [[0, 1, 2, 3],
        [4, 5, 6, 7]],
    [[0, 1, 2, 3],
        [4, 5, 6, 7]],
    [[0, 1, 2, 3],
        [4, 5, 6, 7]]])
print(f"Array dim: {arr.ndim} - size: {arr.size}")
print(f"Array shape: {arr.shape}")

a = np.arange(0, 6, 1)
print(f"before reshape:\n{a}")
b = a.reshape(3, 2)
print(f"after reshape:\n{b}")
b = np.reshape(a, newshape=(1, 6), order='C')
print(f"after reshape:\n{b}")

# 1.4. convert
a = np.arange(0, 6, 1)
print(f"before convert:\n{a}")
print(f"shape before convert:\n{a.shape}")
a2 = a[np.newaxis, :]
a3 = a[:, np.newaxis]
print(f"shape a2:\n{a2.shape}")
print(f"shape a3:\n{a3.shape}")
b2 = np.expand_dims(a, axis=0)
b3 = np.expand_dims(a, axis=1)
print(f"shape b2:\n{b2.shape}")
print(f"shape b3:\n{b3.shape}")

# 1.5. index/slicing
a = np.arange(1, 4, 1)
print(f"slice: {a[-1:]}")

# 1.6. basic array operations

