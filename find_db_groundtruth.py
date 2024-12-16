import numpy as np
import diskannpy as dap
import os
import sys

def main(input_bvecs_file, output_neighbors_file):
    # Verify the input file
    if not os.path.isfile(input_bvecs_file):
        raise FileNotFoundError(f"File {input_bvecs_file} does not exist.")

    # Parameters
    num_vectors = 50000
    dim = 128

    # Load the first 50,000 vectors from the bvecs file
    print("Loading vectors from bvecs file...")
    data = np.empty((num_vectors, dim), dtype=np.uint8)
    with open(input_bvecs_file, "rb") as f:
        for i in range(num_vectors):
            f.read(4)  # Skip the first 4 bytes (size of the vector)
            data[i] = np.frombuffer(f.read(dim), dtype=np.uint8)

    print("Loaded first 50,000 vectors.")

    # DiskANNpy parameters
    index_path = "diskann_index"
    graph_degree = 64
    search_list_size = 1200  # Ensure search list size is greater than k
    num_threads = 12

    # Build the DiskANNpy index
    print("Building DiskANNpy index...")
    dap.build_memory_index(  # Build in-memory index first
        data=data,
        distance_metric="l2",
        index_directory=index_path,
        complexity=search_list_size,
        graph_degree=graph_degree,
        num_threads=num_threads
    )

    print("Index built successfully.")

    # Load the index for querying
    print("Loading index for querying...")
    index = dap.StaticMemoryIndex(
        index_directory=index_path, 
        num_threads=num_threads,
        initial_search_complexity=search_list_size
    )

    # Query each vector for its 1000 nearest neighbors
    print("Querying the index...")
    all_neighbors = []
    response = index.search(query=data[1], k_neighbors=1000, complexity=search_list_size)
    all_neighbors.append(response.identifiers)

    # Convert to numpy array
    all_neighbors = np.array(all_neighbors, dtype=np.int32)

    # Save the neighbors to the output file
    np.save(output_neighbors_file, all_neighbors)
    np.save("testset.npy", data)

    print(f"Neighbor IDs saved to {output_neighbors_file}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_bvecs_file> <output_neighbors_file>")
        sys.exit(1)

    input_bvecs_file = sys.argv[1]
    output_neighbors_file = sys.argv[2]

    main(input_bvecs_file, output_neighbors_file)
  
