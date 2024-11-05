import multiprocessing as mp
import numpy as np
import json
import time
import tqdm
import random
import diskannpy

from scipy.stats import poisson

def send_loop(connection, vectors, wait):
    BATCH_SIZE = 1024
    batches = [vectors[i:i + BATCH_SIZE] for i in range(0, len(vectors), BATCH_SIZE)]
    waits = poisson.rvs(wait*100000, size=len(batches))

    connection.send((len(batches), len(vectors)))
    for b, wait_time in zip(batches, waits):
        connection.send((time.time(), b))
        time.sleep(wait_time / 100000)

def process(batch, idx, cpubudget):
    return idx.batch_search(batch, k_neighbors = 10, complexity = cpubudget, num_threads = 8)

def main(wait, idx, cpubudget):
    (read, prod) = mp.Pipe(duplex = False)
    latencies = []
    test_vecs = np.fromfile("/mnt/nfs/shared/datasets/sift/siftbig/bigann_query.bvecs", dtype = np.uint8).reshape(-1, 132)
    test_vecs = test_vecs[:,4:]

    ids = []
    looper = mp.Process(target=send_loop, args = (prod, test_vecs, wait))
    looper.start()
    start_t = time.time()

    (amt_batch, amt_vecs) = read.recv()

    for i in range(amt_batch):
        (time_batch, data_batch) = read.recv()
        found = process(data_batch, idx, cpubudget)
        latencies.append(time.time() - time_batch)
        #print(latencies[-1])
        ids.append(found.identifiers)

    looper.join()
    #return ids[0]
    return {'throughput' : amt_vecs / (time.time() - start_t), 'latency' : np.mean(latencies)}

def krecall(m1, gnd):
    fnd, exp = 0, 0
    for i in range(m1.shape[0]):
        cpt = m1[i]
        ggn = gnd[i, :len(cpt)]
        inter = set(cpt).intersection(set(ggn))
        fnd += len(inter)
        exp += len(cpt)
    print(fnd/exp)

for membudget in [15]:
    idx = diskannpy.StaticDiskIndex(index_directory = "/mnt/nfs/shared/datasets/sift/100M", num_nodes_to_cache = 256, num_threads = 16, distance_metric = "l2", vector_dtype = np.uint8, dimensions = 128, index_prefix = f"idx400_{membudget}")
    for cpubudget in [26]:

        results = dict()
        for i in tqdm.tqdm(np.linspace(10.0, 0.000025, 10)):
            results[i] = main(i, idx, cpubudget)
            #fnd = results[i]
            #gnd = np.fromfile("gnd/idx_100M.ivecs", dtype = np.uint32).reshape(-1, 1001)[:, 1:]
            #print(krecall(fnd, gnd))

        print(results)
        path = "/mnt/nfs/shared/siftlogs/"
        with open(f"{path}{membudget}GB-{cpubudget}complex.json", "w") as f:
             json.dump(results, f, indent = 4)
