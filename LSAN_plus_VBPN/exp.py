import random
import numpy

numpy.random.seed(42)

for i in range(12):
    std = numpy.random.uniform(0.001, 0.005)
    rand_val = numpy.random.normal(size=(32, 32, 4))

    print(std)
    print(rand_val.sum())