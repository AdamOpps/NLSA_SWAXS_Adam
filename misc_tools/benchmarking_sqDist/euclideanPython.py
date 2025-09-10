import numpy as np
import time

def squared_euclidean_distance(point1, point2):
    return np.sum((point1 - point2)**2)

def main():
    # Define ranges and steps
    min_N, max_N, step_N = 1000, 10000, 1000
    min_D, max_D, step_D = 1000, 10000, 1000

    # Iterate over different values of N
    for N in range(min_N, max_N + 1, step_N):
        # Iterate over different values of D
        for D in range(min_D, max_D + 1, step_D):
            # Generate random data for xyz
            xyz = np.random.rand(N, D)

            # Allocate memory for the squared Euclidean distances
            sq_dist = np.zeros((N, N))

            # Start measuring runtime
            start_time = time.time()

            # Calculate squared Euclidean distances and store the result
            for i in range(N):
                for j in range(N):
                    sq_dist[i, j] = squared_euclidean_distance(xyz[i], xyz[j])

            # Stop measuring runtime
            end_time = time.time()

            # Print runtime and dimensions
            print(f'N={N}, D={D}, Runtime: {end_time - start_time} seconds')

if __name__ == "__main__":
    main()
