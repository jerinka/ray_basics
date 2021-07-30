import ray
import time

ray.init()

#@ray.remote
def f(i):
    k = 0
    for j in range(10000000):
        k+=j
    return k

if 0:
    start = time.time()
    futures = [f.remote(i) for i in range(4)]
    print(ray.get(futures))
    print('ray time = ',time.time()-start)

else:
    start = time.time()
    futures = [f(i) for i in range(1)]
    print(futures)
    print('python time = ',time.time()-start)
