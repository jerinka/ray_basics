import ray
ray.init()

y = 1
obj_ref = ray.put(y)
assert ray.get(obj_ref) == 1

