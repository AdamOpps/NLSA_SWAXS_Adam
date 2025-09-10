#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

double squaredEuclideanDistance(double* point1, double* point2, int D) {
    double distance = 0.0;
    #pragma omp simd reduction(+:distance)
    for (int i = 0; i < D; i++) {
        double diff = point1[i] - point2[i];
        distance += diff * diff;
    }
    return distance;
}

double get_wall_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

int main() {
    // Define ranges and steps
    int min_N = 1000;
    int max_N = 10000;
    int step_N = 1000;

    int min_D = 1000;
    int max_D = 10000;
    int step_D = 1000;

    // Iterate over different values of N
    for (int N = min_N; N <= max_N; N += step_N) {
        // Iterate over different values of D
        for (int D = min_D; D <= max_D; D += step_D) {
            // Allocate memory for the 'xyz' array and fill it with random values
            double** xyz = (double**)malloc(N * sizeof(double*));
            for (int i = 0; i < N; i++) {
                xyz[i] = (double*)malloc(D * sizeof(double));
                for (int j = 0; j < D; j++) {
                    xyz[i][j] = (double)rand() / RAND_MAX;
                }
            }

            // Allocate memory for the squared Euclidean distances
            double** sqDist = (double**)malloc(N * sizeof(double*));
            for (int i = 0; i < N; i++) {
                sqDist[i] = (double*)malloc(N * sizeof(double));
            }

            // Start measuring runtime
            //clock_t start_time = clock();
            double start_time = get_wall_time();

            // Calculate squared Euclidean distances and store the result
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    sqDist[i][j] = squaredEuclideanDistance(xyz[i], xyz[j], D);
                }
            }

            // Stop measuring runtime
            //clock_t end_time = clock();
            double end_time = get_wall_time();

            // Print runtime and dimensions
            //printf("N=%d, D=%d, Runtime: %f seconds\n", N, D, (double)(end_time - start_time) / CLOCKS_PER_SEC);
            printf("N=%d, D=%d, Runtime: %f seconds\n", N, D, end_time - start_time);

            // Free allocated memory
            for (int i = 0; i < N; i++) {
                free(xyz[i]);
                free(sqDist[i]);
            }
            free(xyz);
            free(sqDist);
        }
    }

    return 0;
}
