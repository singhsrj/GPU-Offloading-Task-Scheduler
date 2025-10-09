/*
nvcc attempt2.cu -o a -arch=sm_86
./a
*/


#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>
#include <omp.h>
#include <algorithm>

#define MAX_N 1024

using namespace std;
using namespace std::chrono;

// ==================== Execution Tags ====================
enum class ExecutionTag {
    CPU,
    GPU,
    AUTO
};

// ==================== GPU Kernels (Your Original Code) ====================
__global__ void vectorAddKernel(int *a, int *b, int *c, int n) {
    //  printf("line24");
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

// ==================== CPU Implementations ====================
void vectorAddCPU(vector<int>& A, vector<int>& B, vector<int>& C) {
    int n = A.size();
   
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

void matMulCPU(vector<int>& A, vector<int>& B, vector<int>& C, int N) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            int sum = 0;
            for(int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void lcsCPU(string X, string Y, vector<int>& dp) {
    int m = X.size(), n = Y.size();
    
    // Initialize DP table
    for(int i = 0; i <= m; i++) {
        for(int j = 0; j <= n; j++) {
            dp[i*(n+1) + j] = 0;
        }
    }
    
    // Fill DP table
    for(int i = 1; i <= m; i++) {
        for(int j = 1; j <= n; j++) {
            if(X[i-1] == Y[j-1]) {
                dp[i*(n+1) + j] = dp[(i-1)*(n+1) + (j-1)] + 1;
            } else {
                dp[i*(n+1) + j] = max(dp[(i-1)*(n+1) + j], dp[i*(n+1) + (j-1)]);
            }
        }
    }
}

// ==================== GPU Implementations (Your Original Code) ====================
void vectorAddGPU(vector<int>& A, vector<int>& B, vector<int>& C){
     std::cout<<"line98";
    int n = A.size();
    int *d_a, *d_b, *d_c;
    size_t size = n*sizeof(int);

    cudaMalloc(&d_a, size);
    std::cout<<"line104"<<endl;
    cudaMalloc(&d_b, size);
    std::cout<<"line106"<<endl;
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, A.data(), size, cudaMemcpyHostToDevice);
    std::cout<<"line106"<<endl;
    cudaMemcpy(d_b, B.data(), size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1)/threads;
    
    vectorAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout<<"line121"<<endl;
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for GPU to finish
    std::cout<<"line106"<<endl;

    cudaDeviceSynchronize();

    std::cout<<"line106"<<endl;

    cudaMemcpy(C.data(), d_c, size, cudaMemcpyDeviceToHost);
    std::cout<<"line106"<<endl;


    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    std::cout<<"line106--------------"<<endl;

}

void matMulGPU(vector<int>& A, vector<int>& B, vector<int>& C, int N){
    int *d_A, *d_B, *d_C;
    size_t size = N*N*sizeof(int);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N+15)/16, (N+15)/16);

    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void lcsGPU(string X, string Y, vector<int>& dp){
    int m = X.size(), n = Y.size();
    char *d_X, *d_Y;
    int *d_dp;

    cudaMalloc(&d_X, m*sizeof(char));
    cudaMalloc(&d_Y, n*sizeof(char));
    cudaMalloc(&d_dp, (m+1)*(n+1)*sizeof(int));

    cudaMemcpy(d_X, X.c_str(), m*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y.c_str(), n*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_dp, 0, (m+1)*(n+1)*sizeof(int));

    dim3 threads(16,16);
    dim3 blocks((n+15)/16, (m+15)/16);

    lcsKernel<<<blocks, threads>>>(d_X,d_Y,d_dp,m,n);

    cudaMemcpy(dp.data(), d_dp, (m+1)*(n+1)*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_dp);
}

// ==================== Scheduler Class ====================
class TaskScheduler {
private:
    bool gpu_available;

public:
    TaskScheduler() {
        // Check if GPU is available
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        gpu_available = (error == cudaSuccess && device_count > 0);
        
        if(gpu_available) {
            cout << "GPU detected: Scheduler initialized with GPU support\n";
        } else {
            cout << "No GPU detected: CPU-only mode\n";
        }
    }

    // ==================== Unified Interface Functions ====================
    
    void vectorAdd(ExecutionTag tag, vector<int>& A, vector<int>& B, vector<int>& C) {
        auto start = high_resolution_clock::now();
        
        if(tag == ExecutionTag::GPU && gpu_available) {
            cout << "Executing Vector Addition on GPU...\n";
            vectorAddGPU(A, B, C);
        } else if(tag == ExecutionTag::CPU || !gpu_available) {
            cout << "Executing Vector Addition on CPU...\n";
            vectorAddCPU(A, B, C);
        } else if(tag == ExecutionTag::AUTO) {
            // Simple heuristic: use GPU for large vectors
            if(A.size() > 1000 && gpu_available) {
                cout << "AUTO: Executing Vector Addition on GPU (large size)...\n";
                vectorAddGPU(A, B, C);
            } else {
                cout << "AUTO: Executing Vector Addition on CPU (small size)...\n";
                vectorAddCPU(A, B, C);
            }
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "Execution time: " << duration.count() << " microseconds\n\n";
    }

    void matrixMultiply(ExecutionTag tag, vector<int>& A, vector<int>& B, vector<int>& C, int N) {
        auto start = high_resolution_clock::now();
        
        if(tag == ExecutionTag::GPU && gpu_available) {
            cout << "Executing Matrix Multiplication on GPU...\n";
            matMulGPU(A, B, C, N);
        } else if(tag == ExecutionTag::CPU || !gpu_available) {
            cout << "Executing Matrix Multiplication on CPU...\n";
            matMulCPU(A, B, C, N);
        } else if(tag == ExecutionTag::AUTO) {
            // Better heuristic: use GPU for matrices larger than 128x128
            // GPU memory transfer overhead makes small matrices slower on GPU
            if(N > 128 && gpu_available) {
                cout << "AUTO: Executing Matrix Multiplication on GPU (" << N << "x" << N << ")...\n";
                matMulGPU(A, B, C, N);
            } else {
                cout << "AUTO: Executing Matrix Multiplication on CPU (" << N << "x" << N << ")...\n";
                matMulCPU(A, B, C, N);
            }
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "Execution time: " << duration.count() << " microseconds\n\n";
    }

    void longestCommonSubsequence(ExecutionTag tag, string X, string Y, vector<int>& dp) {
        auto start = high_resolution_clock::now();
        
        if(tag == ExecutionTag::GPU && gpu_available) {
            cout << "Executing LCS on GPU...\n";
            lcsGPU(X, Y, dp);
        } else if(tag == ExecutionTag::CPU || !gpu_available) {
            cout << "Executing LCS on CPU...\n";
            lcsCPU(X, Y, dp);
        } else if(tag == ExecutionTag::AUTO) {
            // Simple heuristic: use GPU for long strings
            if(X.size() * Y.size() > 10000 && gpu_available) {
                cout << "AUTO: Executing LCS on GPU (large problem)...\n";
                lcsGPU(X, Y, dp);
            } else {
                cout << "AUTO: Executing LCS on CPU (small problem)...\n";
                lcsCPU(X, Y, dp);
            }
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "Execution time: " << duration.count() << " microseconds\n\n";
    }
};

// ==================== Utility Functions ====================
ExecutionTag parseExecutionTag(const string& tag_str) {
    if(tag_str == "cpu" || tag_str == "CPU") return ExecutionTag::CPU;
    else if(tag_str == "gpu" || tag_str == "GPU") return ExecutionTag::GPU;
    else if(tag_str == "auto" || tag_str == "AUTO") return ExecutionTag::AUTO;
    else return ExecutionTag::AUTO; // Default
}

string tagToString(ExecutionTag tag) {
    switch(tag) {
        case ExecutionTag::CPU: return "CPU";
        case ExecutionTag::GPU: return "GPU";
        case ExecutionTag::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

// ==================== Main Function ====================
int main(){
    TaskScheduler scheduler;
    
    int choice;
    string execution_target;
    
    cout << "==== CPU/GPU Task Scheduler ====\n";
    cout << "Available execution tags: CPU, GPU, AUTO\n\n";
    
    cout << "Choose program to run:\n";
    cout << "1. Matrix Multiplication\n2. Vector Addition\n3. LCS (Dynamic Programming)\n";
    cout << "Enter choice: ";
    cin >> choice;
    
    cout << "Enter execution target (CPU/GPU/AUTO): ";
    cin >> execution_target;
    ExecutionTag tag = parseExecutionTag(execution_target);
    
    cout << "\n==== Executing with " << tagToString(tag) << " tag ====\n";

    if(choice == 1) {
        int N =800;
  
        
        vector<int> A(N*N), B(N*N), C(N*N);

        for(int i = 0; i < N; i++) 
        {
            for(int j=0;j<N;j++) {
       A[i*N + j] = i;     // Correct 1D indexing
B[i*N + j] = 3*i;
        }

    }
        
        scheduler.matrixMultiply(tag, A, B, C, N);

        cout << "Result matrix C:\n";
        // for(int i = 0; i < N; i++) {
        //     for(int j = 0; j < N; j++) {
        //         cout << C[i*N + j] << " ";
        //     }
        //     cout << "\n";
        // }
    }
    else if(choice == 2) {
        int n = 1000;

        
        vector<int> a(n), b(n), c(n);

        cout << "Enter vector A elements:\n";
        for(int i = 0; i < n; i++) a[i] = i*i;
        cout << "Enter vector B elements:\n";
        for(int i = 0; i < n; i++) b[i] = i*i;

        scheduler.vectorAdd(tag, a, b, c);

        cout << "Result vector C:\n";
        for(int i = 0; i < n; i++) cout << c[i] << " ";
        cout << "\n";
    }
    else if(choice == 3) {
        string X, Y;
        cout << "Enter string X: "; cin >> X;
        cout << "Enter string Y: "; cin >> Y;
        
        int m = X.size(), n = Y.size();
        vector<int> dp((m+1)*(n+1), 0);

        scheduler.longestCommonSubsequence(tag, X, Y, dp);

        cout << "Length of LCS: " << dp[m*(n+1) + n] << "\n";
        
        // Optional: Print DP table for small inputs
        if(m <= 10 && n <= 10) {
            cout << "\nDP Table:\n";
            for(int i = 0; i <= m; i++) {
                for(int j = 0; j <= n; j++) {
                    cout << dp[i*(n+1) + j] << " ";
                }
                cout << "\n";
            }
        }
    }
    else {
        cout << "Invalid choice!\n";
    }

    return 0;
}