#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// ==========================================
// 0. CUDA Error Checking Macro
// ==========================================
#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_KERNEL() \
{ \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("Kernel Launch Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
    CHECK(cudaDeviceSynchronize()); \
}

// ==========================================
// 1. Hyperparameters & Network Architecture
// ==========================================
#define INPUT_SIZE 784   // 28x28 pixels
#define HIDDEN_SIZE 256  // Nodes in Hidden Layer
#define OUTPUT_SIZE 10   // Classes (Fashion-MNIST)
#define TILE_SIZE 16     // For Shared Memory
#define BATCH_SIZE 256   // Mini-batch size
#define EPOCHS 100       // จำนวนรอบในการเทรน
#define LEARNING_RATE 0.05f // อัตราการเรียนรู้

// ==========================================
// 2. Data Loading Functions (IDX Format)
// ==========================================
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_images(string filename, float*& images, int& num_images) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_images, sizeof(num_images));
        num_images = reverseInt(num_images);
        file.read((char*)&n_rows, sizeof(n_rows)); file.read((char*)&n_cols, sizeof(n_cols));
        
        int image_size = INPUT_SIZE;
        images = new float[num_images * image_size];
        unsigned char* temp = new unsigned char[num_images * image_size];
        file.read((char*)temp, num_images * image_size);
        for (int i = 0; i < num_images * image_size; i++) images[i] = (float)temp[i] / 255.0f;
        delete[] temp;
    } else { cout << "Cannot open file: " << filename << endl; exit(1); }
}

void read_mnist_labels(string filename, float*& labels, int& num_labels) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_labels, sizeof(num_labels));
        num_labels = reverseInt(num_labels);
        
        labels = new float[num_labels];
        unsigned char* temp = new unsigned char[num_labels];
        file.read((char*)temp, num_labels);
        for (int i = 0; i < num_labels; i++) labels[i] = (float)temp[i];
        delete[] temp;
    } else { cout << "Cannot open file: " << filename << endl; exit(1); }
}

// ==========================================
// 3. CUDA Kernels
// ==========================================
__global__ void matrixMulTiledKernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty, col = bx * TILE_SIZE + tx;
    float Cvalue = 0.0;
    for (int p = 0; p < (K - 1) / TILE_SIZE + 1; ++p) {
        if (row < M && p * TILE_SIZE + tx < K) ds_A[ty][tx] = A[row * K + p * TILE_SIZE + tx];
        else ds_A[ty][tx] = 0.0;
        if (p * TILE_SIZE + ty < K && col < N) ds_B[ty][tx] = B[(p * TILE_SIZE + ty) * N + col];
        else ds_B[ty][tx] = 0.0;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i) Cvalue += ds_A[ty][i] * ds_B[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = Cvalue;
}

__global__ void reluKernel(float* d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_out[idx] = fmaxf(0.0f, d_out[idx]);
}

__global__ void softmaxKernel(float* d_out, int batch_size, int num_classes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size) {
        float max_val = d_out[row * num_classes];
        for (int i = 1; i < num_classes; i++) max_val = fmaxf(max_val, d_out[row * num_classes + i]);
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            d_out[row * num_classes + i] = expf(d_out[row * num_classes + i] - max_val);
            sum += d_out[row * num_classes + i];
        }
        for (int i = 0; i < num_classes; i++) d_out[row * num_classes + i] /= sum;
    }
}

__global__ void evaluate_loss_acc(float* d_O, float* d_Y, float* d_loss, int* d_correct, int batch_size, int num_classes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size) {
        int best_class = 0;
        float max_prob = d_O[row * num_classes];
        float prob_correct = 1e-7; 
        
        for (int i = 0; i < num_classes; i++) {
            float p = d_O[row * num_classes + i];
            if (p > max_prob) { max_prob = p; best_class = i; }
            if (i == (int)d_Y[row]) prob_correct = fmaxf(p, 1e-7); 
        }
        
        if (best_class == (int)d_Y[row]) atomicAdd(d_correct, 1);
        atomicAdd(d_loss, -logf(prob_correct)); 
    }
}

__global__ void compute_dz2(float* d_O, float* d_Y, float* dz2, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_classes) {
        int row = idx / num_classes;
        int col = idx % num_classes;
        float y_val = (d_Y[row] == col) ? 1.0f : 0.0f;
        dz2[idx] = (d_O[idx] - y_val) / batch_size;
    }
}

__global__ void matMulTransposeA(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[i * M + row] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matMulTransposeB(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[col * K + i];
        C[row * N + col] = sum;
    }
}

__global__ void relu_backward(float* dH, float* H, float* dZ1, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dZ1[idx] = (H[idx] > 0.0f) ? dH[idx] : 0.0f;
}

__global__ void update_weights(float* W, float* dW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) W[idx] -= lr * dW[idx];
}

// ==========================================
// 4. Main Training Loop
// ==========================================
int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) { cout << "ERROR: No CUDA devices found!" << endl; return 1; }

    float *h_train_X, *h_train_Y, *h_test_X, *h_test_Y;
    int total_train_size, train_label_size, test_size, test_label_size;

    cout << "Loading datasets..." << endl;
    read_mnist_images("data/train-images-idx3-ubyte", h_train_X, total_train_size);
    read_mnist_labels("data/train-labels-idx1-ubyte", h_train_Y, train_label_size);
    read_mnist_images("data/t10k-images-idx3-ubyte", h_test_X, test_size);
    read_mnist_labels("data/t10k-labels-idx1-ubyte", h_test_Y, test_label_size);
    if(total_train_size == 0 || test_size == 0 || total_train_size != train_label_size) return 1;
    
    // --- Data Splitting ---
    int train_samples = 50000;
    int val_samples = total_train_size - train_samples;
    
    int train_batches = train_samples / BATCH_SIZE;
    int val_batches = val_samples / BATCH_SIZE;
    int test_batches = test_size / BATCH_SIZE;

    cout << "Train Size: " << train_samples << " | Val Size: " << val_samples << " | Test Size: " << test_size << endl;

    float *h_W1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *h_W2 = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) h_W1[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / INPUT_SIZE);
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) h_W2[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / HIDDEN_SIZE);

    float *d_X, *d_Y, *d_W1, *d_W2, *d_H, *d_O;
    float *d_dW1, *d_dW2, *d_dZ1, *d_dZ2, *d_dH;
    int *d_correct;
    float *d_loss; 

    CHECK(cudaMalloc(&d_X, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_Y, BATCH_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_H, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_O, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_dW1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_dW2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_dZ1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_dZ2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_dH, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_correct, sizeof(int)));
    CHECK(cudaMalloc(&d_loss, sizeof(float)));

    CHECK(cudaMemcpy(d_W1, h_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W2, h_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cout << "\nStarting Training..." << endl;
    
    // ตัวแปรสะสมเวลาและเก็บค่าสุดท้าย
    double total_train_time = 0.0;
    float final_train_acc = 0.0f, final_val_acc = 0.0f, final_test_acc = 0.0f;
    float final_train_loss = 0.0f, final_val_loss = 0.0f, final_test_loss = 0.0f;

    // ==========================================
    // --- EPOCH LOOP (Train & Validation) ---
    // ==========================================
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        
        // >>> ⏱️ เริ่มจับเวลาเฉพาะช่วง TRAIN PHASE <<<
        auto train_start = chrono::high_resolution_clock::now();

        // --- 1. TRAIN PHASE ---
        int correct_train = 0;
        float train_loss_total = 0.0f;
        CHECK(cudaMemset(d_correct, 0, sizeof(int)));
        CHECK(cudaMemset(d_loss, 0, sizeof(float)));

        for (int b = 0; b < train_batches; b++) {
            CHECK(cudaMemcpy(d_X, h_train_X + b * BATCH_SIZE * INPUT_SIZE, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_Y, h_train_Y + b * BATCH_SIZE, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            dim3 dimGrid1((HIDDEN_SIZE - 1) / TILE_SIZE + 1, (BATCH_SIZE - 1) / TILE_SIZE + 1, 1);
            dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
            matrixMulTiledKernel<<<dimGrid1, dimBlock>>>(d_X, d_W1, d_H, BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
            reluKernel<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(d_H, BATCH_SIZE * HIDDEN_SIZE);
            
            dim3 dimGrid2((OUTPUT_SIZE - 1) / TILE_SIZE + 1, (BATCH_SIZE - 1) / TILE_SIZE + 1, 1);
            matrixMulTiledKernel<<<dimGrid2, dimBlock>>>(d_H, d_W2, d_O, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
            softmaxKernel<<<(BATCH_SIZE + 255) / 256, 256>>>(d_O, BATCH_SIZE, OUTPUT_SIZE);

            evaluate_loss_acc<<<(BATCH_SIZE + 255) / 256, 256>>>(d_O, d_Y, d_loss, d_correct, BATCH_SIZE, OUTPUT_SIZE);

            compute_dz2<<<(BATCH_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_O, d_Y, d_dZ2, BATCH_SIZE, OUTPUT_SIZE);
            dim3 gridW2((OUTPUT_SIZE - 1) / 16 + 1, (HIDDEN_SIZE - 1) / 16 + 1);
            dim3 blockW2(16, 16);
            matMulTransposeA<<<gridW2, blockW2>>>(d_H, d_dZ2, d_dW2, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
            dim3 gridH((HIDDEN_SIZE - 1) / 16 + 1, (BATCH_SIZE - 1) / 16 + 1);
            matMulTransposeB<<<gridH, blockW2>>>(d_dZ2, d_W2, d_dH, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            relu_backward<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(d_dH, d_H, d_dZ1, BATCH_SIZE * HIDDEN_SIZE);
            dim3 gridW1((HIDDEN_SIZE - 1) / 16 + 1, (INPUT_SIZE - 1) / 16 + 1);
            matMulTransposeA<<<gridW1, blockW2>>>(d_X, d_dZ1, d_dW1, INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE);

            update_weights<<<(INPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(d_W1, d_dW1, LEARNING_RATE, INPUT_SIZE * HIDDEN_SIZE);
            update_weights<<<(HIDDEN_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_W2, d_dW2, LEARNING_RATE, HIDDEN_SIZE * OUTPUT_SIZE);
            CHECK_KERNEL(); 
        }
        
        // >>> ⏱️ หยุดจับเวลา และบวกสะสม <<<
        auto train_end = chrono::high_resolution_clock::now();
        chrono::duration<double> epoch_time = train_end - train_start;
        total_train_time += epoch_time.count();

        // คำนวณ Accuracy & Loss ฝั่ง Train
        CHECK(cudaMemcpy(&correct_train, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&train_loss_total, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        final_train_acc = (float)correct_train / (train_batches * BATCH_SIZE) * 100.0f;
        final_train_loss = train_loss_total / (train_batches * BATCH_SIZE);

        // --- 2. VALIDATION PHASE ---
        int correct_val = 0;
        float val_loss_total = 0.0f;
        CHECK(cudaMemset(d_correct, 0, sizeof(int)));
        CHECK(cudaMemset(d_loss, 0, sizeof(float)));

        for (int b = 0; b < val_batches; b++) {
            CHECK(cudaMemcpy(d_X, h_train_X + (train_samples + b * BATCH_SIZE) * INPUT_SIZE, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_Y, h_train_Y + train_samples + b * BATCH_SIZE, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            dim3 dimGrid1((HIDDEN_SIZE - 1) / TILE_SIZE + 1, (BATCH_SIZE - 1) / TILE_SIZE + 1, 1);
            dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
            matrixMulTiledKernel<<<dimGrid1, dimBlock>>>(d_X, d_W1, d_H, BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
            reluKernel<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(d_H, BATCH_SIZE * HIDDEN_SIZE);
            
            dim3 dimGrid2((OUTPUT_SIZE - 1) / TILE_SIZE + 1, (BATCH_SIZE - 1) / TILE_SIZE + 1, 1);
            matrixMulTiledKernel<<<dimGrid2, dimBlock>>>(d_H, d_W2, d_O, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
            softmaxKernel<<<(BATCH_SIZE + 255) / 256, 256>>>(d_O, BATCH_SIZE, OUTPUT_SIZE);

            evaluate_loss_acc<<<(BATCH_SIZE + 255) / 256, 256>>>(d_O, d_Y, d_loss, d_correct, BATCH_SIZE, OUTPUT_SIZE);
            CHECK_KERNEL();
        }
        CHECK(cudaMemcpy(&correct_val, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&val_loss_total, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        final_val_acc = (float)correct_val / (val_batches * BATCH_SIZE) * 100.0f;
        final_val_loss = val_loss_total / (val_batches * BATCH_SIZE);

        printf("Epoch [%d/%d] - Train Acc: %.3f %% - Val Acc: %.3f %% - Train Loss: %.4f - Val Loss: %.4f\n",
               epoch + 1, EPOCHS, final_train_acc, final_val_acc, final_train_loss, final_val_loss);
    }

    // ==========================================
    // --- 3. FINAL TEST PHASE ---
    // ==========================================
    int correct_test = 0;
    float test_loss_total = 0.0f;
    CHECK(cudaMemset(d_correct, 0, sizeof(int)));
    CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    for (int b = 0; b < test_batches; b++) {
        CHECK(cudaMemcpy(d_X, h_test_X + b * BATCH_SIZE * INPUT_SIZE, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Y, h_test_Y + b * BATCH_SIZE, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        dim3 dimGrid1((HIDDEN_SIZE - 1) / TILE_SIZE + 1, (BATCH_SIZE - 1) / TILE_SIZE + 1, 1);
        dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
        matrixMulTiledKernel<<<dimGrid1, dimBlock>>>(d_X, d_W1, d_H, BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
        reluKernel<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(d_H, BATCH_SIZE * HIDDEN_SIZE);
        
        dim3 dimGrid2((OUTPUT_SIZE - 1) / TILE_SIZE + 1, (BATCH_SIZE - 1) / TILE_SIZE + 1, 1);
        matrixMulTiledKernel<<<dimGrid2, dimBlock>>>(d_H, d_W2, d_O, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
        softmaxKernel<<<(BATCH_SIZE + 255) / 256, 256>>>(d_O, BATCH_SIZE, OUTPUT_SIZE);

        evaluate_loss_acc<<<(BATCH_SIZE + 255) / 256, 256>>>(d_O, d_Y, d_loss, d_correct, BATCH_SIZE, OUTPUT_SIZE);
        CHECK_KERNEL();
    }
    CHECK(cudaMemcpy(&correct_test, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&test_loss_total, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    final_test_acc = (float)correct_test / (test_batches * BATCH_SIZE) * 100.0f;
    final_test_loss = test_loss_total / (test_batches * BATCH_SIZE);

    // --- GFLOPS Calculation (ใช้ total_train_time) ---
    double ops_per_batch = 2.0 * BATCH_SIZE * HIDDEN_SIZE * INPUT_SIZE +
                           2.0 * BATCH_SIZE * OUTPUT_SIZE * HIDDEN_SIZE +
                           2.0 * HIDDEN_SIZE * OUTPUT_SIZE * BATCH_SIZE +
                           2.0 * BATCH_SIZE * HIDDEN_SIZE * OUTPUT_SIZE +
                           2.0 * INPUT_SIZE * HIDDEN_SIZE * BATCH_SIZE;
                           
    double total_ops = ops_per_batch * train_batches * EPOCHS;
    double gflops = (total_ops / total_train_time) / 1e9;

    // --- FINAL OUTPUT ---
    cout << "\n===============================================" << endl;
    cout << "                 FINAL RESULTS                 " << endl;
    cout << "===============================================" << endl;
    printf("%-15s | %-14s | %-10s\n", "Dataset Phase", "Accuracy (%)", "Loss");
    cout << "-----------------------------------------------" << endl;
    printf("%-15s | %-14.3f | %-10.4f\n", "Training", final_train_acc, final_train_loss);
    printf("%-15s | %-14.3f | %-10.4f\n", "Validation", final_val_acc, final_val_loss);
    printf("%-15s | %-14.3f | %-10.4f\n", "Testing", final_test_acc, final_test_loss);
    cout << "-----------------------------------------------" << endl;
    printf("%-20s : %.4f seconds\n", "Total Training Time", total_train_time);
    printf("%-20s : %.3f GFLOPS\n", "Throughput", gflops);
    cout << "===============================================" << endl;

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_H); cudaFree(d_O);
    cudaFree(d_dW1); cudaFree(d_dW2); cudaFree(d_dZ1); cudaFree(d_dZ2); cudaFree(d_dH); cudaFree(d_correct); cudaFree(d_loss);
    delete[] h_train_X; delete[] h_train_Y; delete[] h_test_X; delete[] h_test_Y;
    delete[] h_W1; delete[] h_W2;

    return 0;
}
