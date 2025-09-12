#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>

#define MAX_N 1024

// Vector Addition Kernel
__global__ void vectorAddKernel(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void matMulKernel(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N) {
        int sum = 0;
        for(int k=0; k<N; k++)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

// DP Kernel for LCS (Row by Row)
__global__ void lcsKernel(char *X, char *Y, int *dp, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>0 && j>0 && i<=m && j<=n){
        if(X[i-1] == Y[j-1])
            dp[i*(n+1)+j] = dp[(i-1)*(n+1)+(j-1)] + 1;
        else
            dp[i*(n+1)+j] = max(dp[(i-1)*(n+1)+j], dp[i*(n+1)+(j-1)]);
    }
}

// ==================== GPU Functions ====================

void vectorAddGPU(int *a, int *b, int *c, int n){
    int *d_a, *d_b, *d_c;
    size_t size = n * sizeof(int);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1)/threads;
    vectorAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void matMulGPU(int *A, int *B, int *C, int N){
    int *d_A, *d_B, *d_C;
    size_t size = N*N*sizeof(int);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N+15)/16, (N+15)/16);

    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void lcsGPU(char *X, char *Y, int *dp, int m, int n){
    char *d_X, *d_Y;
    int *d_dp;
    cudaMalloc(&d_X, m*sizeof(char));
    cudaMalloc(&d_Y, n*sizeof(char));
    cudaMalloc(&d_dp, (m+1)*(n+1)*sizeof(int));

    cudaMemcpy(d_X, X, m*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, n*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_dp, 0, (m+1)*(n+1)*sizeof(int));

    dim3 threads(16,16);
    dim3 blocks((n+15)/16, (m+15)/16);

    lcsKernel<<<blocks, threads>>>(d_X,d_Y,d_dp,m,n);

    cudaMemcpy(dp, d_dp, (m+1)*(n+1)*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_dp);
}


int main(){
    int choice;
    printf("Choose program to run on GPU:\n");
    printf("1. Matrix Multiplication\n2. Vector Addition\n3. LCS (DP)\n");
    scanf("%d",&choice);

    if(choice==1){
        int N;
        printf("Enter matrix size N x N: ");
        scanf("%d",&N);
        int *A = (int*)malloc(N*N*sizeof(int));
        int *B = (int*)malloc(N*N*sizeof(int));
        int *C = (int*)malloc(N*N*sizeof(int));

        printf("Enter matrix A elements:\n");
        for(int i=0;i<N*N;i++) scanf("%d",&A[i]);
        printf("Enter matrix B elements:\n");
        for(int i=0;i<N*N;i++) scanf("%d",&B[i]);

        matMulGPU(A,B,C,N);

        printf("Result matrix C:\n");
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++) printf("%d ",C[i*N+j]);
            printf("\n");
        }

        free(A); free(B); free(C);
    }
    else if(choice==2){
        int n;
        printf("Enter vector size: ");
        scanf("%d",&n);
        int *a = (int*)malloc(n*sizeof(int));
        int *b = (int*)malloc(n*sizeof(int));
        int *c = (int*)malloc(n*sizeof(int));

        printf("Enter vector A elements:\n");
        for(int i=0;i<n;i++) scanf("%d",&a[i]);
        printf("Enter vector B elements:\n");
        for(int i=0;i<n;i++) scanf("%d",&b[i]);

        vectorAddGPU(a,b,c,n);

        printf("Result vector C:\n");
        for(int i=0;i<n;i++) printf("%d ",c[i]);
        printf("\n");

        free(a); free(b); free(c);
    }
    else if(choice==3){
        char X[MAX_N], Y[MAX_N];
        printf("Enter string X: "); scanf("%s",X);
        printf("Enter string Y: "); scanf("%s",Y);
        int m = strlen(X), n = strlen(Y);
        int *dp = (int*)malloc((m+1)*(n+1)*sizeof(int));

        lcsGPU(X,Y,dp,m,n);

        printf("Length of LCS: %d\n", dp[m*(n+1)+n]);

        free(dp);
    }
    else printf("Invalid choice!\n");

    return 0;
}
