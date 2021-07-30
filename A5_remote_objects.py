import numpy as np
import ray
import time

ray.init()

def create_matrix1(size):
    return np.random.normal(size=size)

@ray.remote
def create_matrix2(size):
    return np.random.normal(size=size)

@ray.remote
def multiply_matrices(x, y):
    return np.dot(x, y)
start = time.time()
x_id = create_matrix2.remote([1000, 1000])
y_id1 = create_matrix1([1000, 1000])
y_id = ray.put(y_id1)

z_id = multiply_matrices.remote(x_id, y_id)

# Get the results.
z = ray.get(z_id)
print('z:',z)
print('Time=',time.time()-start)

