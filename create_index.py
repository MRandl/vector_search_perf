import numpy as np 
import diskannpy

vecs = np.load("/mnt/nfs/shared/datasets/sift/100M/100M.npy")
#diskannpy.build_disk_index(data = vecs, distance_metric = "l2", index_directory = "/mnt/nfs/shared/datasets/sift/siftbig", complexity = 100, graph_degree = 64, search_memory_maximum = 80.0, build_memory_maximum = 400.0, num_threads = 32, vector_dtype = np.uint8, index_prefix = "iddxtest")
diskannpy.build_memory_index(data = vecs, distance_metric = "l2", index_directory = "/mnt/nfs/shared/datasets/sift/100M", complexity = 100, graph_degree = 64, num_threads = 32, alpha = 1.2000000476837158, vector_dtype = np.uint8, index_prefix = "idxmem64_1.2")
