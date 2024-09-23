import multiprocessing as mp
import numpy as np
import time
import tqdm
import random
import diskannpy

def send_loop(connection, vectors, wait):
    BATCH_SIZE = 32
    batches = [vectors[i:i + BATCH_SIZE] for i in range(0, len(vectors), BATCH_SIZE)]
    
    connection.send((len(batches), len(vectors)))
    for b in batches:
        connection.send((time.time(), b))
        time.sleep(wait)

def process(i):
    time.sleep(random.randint(0, 1000) / 10000)
    return i

def main(wait):
    (read, prod) = mp.Pipe(duplex = False)
    latencies = []
    looper = mp.Process(target=send_loop, args = (prod, [999]*5000, wait))
    looper.start()
    start_t = time.time()

    (amt_batch, amt_vecs) = read.recv()
    
    for i in range(amt_batch):
        target = read.recv()
        process(target)
        latencies.append(time.time() - target[0])

    looper.join()

    return {'throughput' : amt_vecs / (time.time() - start_t), 'latency' : np.mean(latencies)}

results = dict()
for i in tqdm.tqdm(np.linspace(0.20, 0.025, 10)):
    results[i] = main(i)

print(results)
