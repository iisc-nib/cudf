#include <stdio.h>

__global__ void print_hello() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Hello from the first thread of the first block!\n");
  }
}

void print_hello_cpp(){
    print_hello<<<1, 1>>>();
    cudaDeviceSynchronize();
}