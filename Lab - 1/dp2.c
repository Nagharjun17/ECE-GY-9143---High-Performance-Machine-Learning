#include<stdio.h>
#include<stdlib.h>
#include <time.h>

float dpunroll(long N, float *pA, float *pB) {
float R = 0.0;
int j;
for (j = 0; j < N; j += 4)
R += pA[j] * pB[j] + pA[j + 1] * pB[j + 1] + pA[j + 2] * pB[j + 2] + pA[j + 3] * pB[j + 3];
return R;
}
int main(int argc, char *argv[]) {
    struct timespec start, end;
    char *k1 = argv[1];
    char *k2 = argv[2];
    long size_of_array = atoi(k1);
    int iterations = atoi(k2);
    float a[size_of_array];
    float b[size_of_array];

    for(long i = 0; i < size_of_array; i++) {
        a[i] = 1;
        b[i] = 1;
    }

    double average_et = 0.0;
    double average_et_per_iteration_in_sec = 0.0;
    double gflops = 0.0;
    double bandwidth = 0.0;
    float p = 0.0;

    for(long i = 0; i < iterations; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        p = dpunroll(size_of_array, a, b);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_usec=(((double)end.tv_sec * 1000000 + (double)end.tv_nsec / 1000) - ((double)start.tv_sec * 1000000 + (double)start.tv_nsec / 1000));
        if(i > (iterations / 2) - 1) {
            average_et = average_et + time_usec;
        }
    }
    average_et_per_iteration_in_sec = average_et / ((float)(iterations / 2) * 1000000);
    bandwidth = 9 * 4 * (size_of_array / 4) / (average_et_per_iteration_in_sec * 1000000000); // 3 load and store and converstion from bytes to gb
    gflops = 8 * (size_of_array / 4) / (average_et_per_iteration_in_sec * 1000000000); // 2 floating point ops

    if((long)p < size_of_array) {
            printf("N: %lu <T>: %.08lf sec B: %0.05lf GB/sec F: %.05lf GFLOPS", size_of_array, average_et_per_iteration_in_sec, bandwidth, gflops);
    } else {
            printf("N: %lu <T>: %.08lf sec B: %0.05lf GB/sec F: %.05lf GFLOPS", (long)p, average_et_per_iteration_in_sec, bandwidth, gflops);
    }
    return 0;
}