import time
import numpy as np
import sys

size_of_array = int(sys.argv[1])
iterations = int(sys.argv[2])

A = np.ones(size_of_array, dtype = np.float32)
B = np.ones(size_of_array, dtype = np.float32)

average_et = 0.0
for k in range(0, iterations):
    start = time.time()
    R = np.dot(A, B)
    end = time.time()
    diff = end - start
    if(k > (iterations / 2) - 1):
        average_et = average_et + diff
average_et = average_et / (float)(iterations / 2)
bandwidth = (3 * 4 * size_of_array / average_et) / 1000000000 # 3 load and store and converstion from bytes to gb
gflops = (2 * size_of_array / average_et) / 1000000000 # 2 floating point ops

print("N: " + str(size_of_array) + " <T>: " + str(average_et) + " sec B: " + str(bandwidth) + " GB/sec F: " + str(gflops) + " GFLOPS")

