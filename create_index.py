import numpy as np 
import diskannpy

#vecs = np.read("/mnt/nfs/shared/datasets/sift/siftbig/betterbigsift_base.npy")
vecs = np.random.randint(0, 256, dtype = np.uint8, size = (128, 1200))
diskannpy.build_index(data = vecs, distance_metric = "l2", index_directory = "/mnt/nfs/shared/datasets/sift/siftbig", complexity = 100, graph_degree = 64, search_memory_maximum = 80.0, build_memory_maximum = 400.0, num_threads = 32, vector_dtype = np.uint8, index_prefix = "iddxtest")
