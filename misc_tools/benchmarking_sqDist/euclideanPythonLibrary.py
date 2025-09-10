import numpy as np
import time
from scipy.spatial.distance import cdist

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

            # Start measuring runtime
            start_time = time.time()

            # Calculate squared Euclidean distances using cdist
            sq_dist = cdist(xyz, xyz, 'sqeuclidean')

            # Stop measuring runtime
            end_time = time.time()

            # Print runtime and dimensions
            print(f'N={N}, D={D}, Runtime: {end_time - start_time} seconds')

if __name__ == "__main__":
    main()
