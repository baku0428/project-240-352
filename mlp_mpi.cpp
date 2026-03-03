#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <cstdlib>

using namespace std;

// ==========================================
// 1. Hyperparameters & Network Architecture
// ==========================================
#define INPUT_SIZE 784   // 28x28 pixels
#define HIDDEN_SIZE 256  // Nodes in Hidden Layer
#define OUTPUT_SIZE 10   // Classes (Fashion-MNIST)
#define BATCH_SIZE 256   // Global Mini-batch size
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
// 3. CPU Math Functions (Replacing CUDA)
// ==========================================
void matmul(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

void matMulTransposeA(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[k * M + i] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

void matMulTransposeB(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[j * K + k];
            C[i * N + j] = sum;
        }
    }
}

void relu(float* H, int size) {
    for (int i = 0; i < size; ++i) H[i] = fmaxf(0.0f, H[i]);
}

void relu_backward(float* dH, float* H, float* dZ1, int size) {
    for (int i = 0; i < size; ++i) dZ1[i] = (H[i] > 0.0f) ? dH[i] : 0.0f;
}

void softmax(float* O, int batch_size, int num_classes) {
    for (int row = 0; row < batch_size; ++row) {
        float max_val = O[row * num_classes];
        for (int i = 1; i < num_classes; i++) max_val = fmaxf(max_val, O[row * num_classes + i]);
        
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            O[row * num_classes + i] = expf(O[row * num_classes + i] - max_val);
            sum += O[row * num_classes + i];
        }
        for (int i = 0; i < num_classes; i++) O[row * num_classes + i] /= sum;
    }
}

void evaluate_loss_acc(float* O, float* Y, float& loss, int& correct, int batch_size, int num_classes) {
    for (int row = 0; row < batch_size; ++row) {
        int best_class = 0;
        float max_prob = O[row * num_classes];
        float prob_correct = 1e-7; 
        
        for (int i = 0; i < num_classes; i++) {
            float p = O[row * num_classes + i];
            if (p > max_prob) { max_prob = p; best_class = i; }
            if (i == (int)Y[row]) prob_correct = fmaxf(p, 1e-7); 
        }
        
        if (best_class == (int)Y[row]) correct++;
        loss += -logf(prob_correct); 
    }
}

void compute_dz2(float* O, float* Y, float* dz2, int batch_size, int global_batch_size, int num_classes) {
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < num_classes; ++col) {
            float y_val = (Y[row] == col) ? 1.0f : 0.0f;
            // หารด้วย Global Batch Size เพื่อให้ Gradient รวมกันได้อย่างถูกต้องในทุก processes
            dz2[row * num_classes + col] = (O[row * num_classes + col] - y_val) / global_batch_size;
        }
    }
}

void update_weights(float* W, float* dW, float lr, int size) {
    for (int i = 0; i < size; ++i) W[i] -= lr * dW[i];
}

// ==========================================
// 4. Main Training Loop
// ==========================================
int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (BATCH_SIZE % size != 0 && rank == 0) {
        cout << "ERROR: BATCH_SIZE (" << BATCH_SIZE << ") must be divisible by number of MPI processes (" << size << ")." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int local_batch_size = BATCH_SIZE / size;

    float *h_train_X, *h_train_Y, *h_test_X, *h_test_Y;
    int total_train_size, train_label_size, test_size, test_label_size;

    if (rank == 0) cout << "Loading datasets..." << endl;
    
    // ทุก Process โหลด Data เพื่อความสะดวก (หรือสามารถให้ Rank 0 โหลดแล้ว Scatter ได้)
    read_mnist_images("data/train-images-idx3-ubyte", h_train_X, total_train_size);
    read_mnist_labels("data/train-labels-idx1-ubyte", h_train_Y, train_label_size);
    read_mnist_images("data/t10k-images-idx3-ubyte", h_test_X, test_size);
    read_mnist_labels("data/t10k-labels-idx1-ubyte", h_test_Y, test_label_size);

    int train_samples = 50000;
    int val_samples = total_train_size - train_samples;
    
    int train_batches = train_samples / BATCH_SIZE;
    int val_batches = val_samples / BATCH_SIZE;
    int test_batches = test_size / BATCH_SIZE;

    if (rank == 0) {
        cout << "Train Size: " << train_samples << " | Val Size: " << val_samples << " | Test Size: " << test_size << endl;
    }

    // Allocate Model Weights
    float *W1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *W2 = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    
    // Allocate Memory for gradients and activations
    float *X = new float[local_batch_size * INPUT_SIZE];
    float *Y = new float[local_batch_size];
    float *H = new float[local_batch_size * HIDDEN_SIZE];
    float *O = new float[local_batch_size * OUTPUT_SIZE];
    float *dW1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *dW2 = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    float *dZ1 = new float[local_batch_size * HIDDEN_SIZE];
    float *dZ2 = new float[local_batch_size * OUTPUT_SIZE];
    float *dH = new float[local_batch_size * HIDDEN_SIZE];

    // Rank 0 สุ่ม Weights และแจกจ่าย (Broadcast) ให้ทุก Ranks
    if (rank == 0) {
        for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) W1[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / INPUT_SIZE);
        for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) W2[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / HIDDEN_SIZE);
    }
    MPI_Bcast(W1, INPUT_SIZE * HIDDEN_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(W2, HIDDEN_SIZE * OUTPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) cout << "\nStarting Distributed Training on " << size << " MPI Processes..." << endl;
    
    double total_train_time = 0.0;
    float final_train_acc = 0.0f, final_val_acc = 0.0f, final_test_acc = 0.0f;
    float final_train_loss = 0.0f, final_val_loss = 0.0f, final_test_loss = 0.0f;

    // ==========================================
    // --- EPOCH LOOP ---
    // ==========================================
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        
        double train_start = MPI_Wtime();

        // --- 1. TRAIN PHASE ---
        int correct_train_local = 0, correct_train_global = 0;
        float train_loss_local = 0.0f, train_loss_global = 0.0f;

        for (int b = 0; b < train_batches; b++) {
            // ดึงข้อมูลเฉพาะส่วนของ Process ปัจจุบัน
            int offset = (b * BATCH_SIZE) + (rank * local_batch_size);
            for(int i=0; i<local_batch_size * INPUT_SIZE; ++i) X[i] = h_train_X[offset * INPUT_SIZE + i];
            for(int i=0; i<local_batch_size; ++i) Y[i] = h_train_Y[offset + i];

            // Forward Pass
            matmul(X, W1, H, local_batch_size, HIDDEN_SIZE, INPUT_SIZE);
            relu(H, local_batch_size * HIDDEN_SIZE);
            
            matmul(H, W2, O, local_batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
            softmax(O, local_batch_size, OUTPUT_SIZE);

            evaluate_loss_acc(O, Y, train_loss_local, correct_train_local, local_batch_size, OUTPUT_SIZE);

            // Backward Pass
            compute_dz2(O, Y, dZ2, local_batch_size, BATCH_SIZE, OUTPUT_SIZE);
            
            matMulTransposeA(H, dZ2, dW2, HIDDEN_SIZE, OUTPUT_SIZE, local_batch_size);
            matMulTransposeB(dZ2, W2, dH, local_batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
            
            relu_backward(dH, H, dZ1, local_batch_size * HIDDEN_SIZE);
            matMulTransposeA(X, dZ1, dW1, INPUT_SIZE, HIDDEN_SIZE, local_batch_size);

            // MPI Allreduce: รวม Gradient จากทุก Process
            MPI_Allreduce(MPI_IN_PLACE, dW1, INPUT_SIZE * HIDDEN_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, dW2, HIDDEN_SIZE * OUTPUT_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            // Update Weights
            update_weights(W1, dW1, LEARNING_RATE, INPUT_SIZE * HIDDEN_SIZE);
            update_weights(W2, dW2, LEARNING_RATE, HIDDEN_SIZE * OUTPUT_SIZE);
        }
        
        double train_end = MPI_Wtime();
        total_train_time += (train_end - train_start);

        // รวม Loss และ Accuracy ของทั้ง Epoch
        MPI_Allreduce(&correct_train_local, &correct_train_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&train_loss_local, &train_loss_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        
        final_train_acc = (float)correct_train_global / (train_batches * BATCH_SIZE) * 100.0f;
        final_train_loss = train_loss_global / (train_batches * BATCH_SIZE);

        // --- 2. VALIDATION PHASE ---
        int correct_val_local = 0, correct_val_global = 0;
        float val_loss_local = 0.0f, val_loss_global = 0.0f;

        for (int b = 0; b < val_batches; b++) {
            int offset = train_samples + (b * BATCH_SIZE) + (rank * local_batch_size);
            for(int i=0; i<local_batch_size * INPUT_SIZE; ++i) X[i] = h_train_X[offset * INPUT_SIZE + i];
            for(int i=0; i<local_batch_size; ++i) Y[i] = h_train_Y[offset + i];

            matmul(X, W1, H, local_batch_size, HIDDEN_SIZE, INPUT_SIZE);
            relu(H, local_batch_size * HIDDEN_SIZE);
            
            matmul(H, W2, O, local_batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
            softmax(O, local_batch_size, OUTPUT_SIZE);

            evaluate_loss_acc(O, Y, val_loss_local, correct_val_local, local_batch_size, OUTPUT_SIZE);
        }
        MPI_Allreduce(&correct_val_local, &correct_val_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&val_loss_local, &val_loss_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        
        final_val_acc = (float)correct_val_global / (val_batches * BATCH_SIZE) * 100.0f;
        final_val_loss = val_loss_global / (val_batches * BATCH_SIZE);

        if (rank == 0) {
            printf("Epoch [%d/%d] - Train Acc: %.3f %% - Val Acc: %.3f %% - Train Loss: %.4f - Val Loss: %.4f\n",
                   epoch + 1, EPOCHS, final_train_acc, final_val_acc, final_train_loss, final_val_loss);
        }
    }

    // ==========================================
    // --- 3. FINAL TEST PHASE ---
    // ==========================================
    int correct_test_local = 0, correct_test_global = 0;
    float test_loss_local = 0.0f, test_loss_global = 0.0f;

    for (int b = 0; b < test_batches; b++) {
        int offset = (b * BATCH_SIZE) + (rank * local_batch_size);
        for(int i=0; i<local_batch_size * INPUT_SIZE; ++i) X[i] = h_test_X[offset * INPUT_SIZE + i];
        for(int i=0; i<local_batch_size; ++i) Y[i] = h_test_Y[offset + i];

        matmul(X, W1, H, local_batch_size, HIDDEN_SIZE, INPUT_SIZE);
        relu(H, local_batch_size * HIDDEN_SIZE);
        
        matmul(H, W2, O, local_batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
        softmax(O, local_batch_size, OUTPUT_SIZE);

        evaluate_loss_acc(O, Y, test_loss_local, correct_test_local, local_batch_size, OUTPUT_SIZE);
    }
    MPI_Allreduce(&correct_test_local, &correct_test_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&test_loss_local, &test_loss_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    final_test_acc = (float)correct_test_global / (test_batches * BATCH_SIZE) * 100.0f;
    final_test_loss = test_loss_global / (test_batches * BATCH_SIZE);

    if (rank == 0) {
        double ops_per_batch = 2.0 * BATCH_SIZE * HIDDEN_SIZE * INPUT_SIZE +
                               2.0 * BATCH_SIZE * OUTPUT_SIZE * HIDDEN_SIZE +
                               2.0 * HIDDEN_SIZE * OUTPUT_SIZE * BATCH_SIZE +
                               2.0 * BATCH_SIZE * HIDDEN_SIZE * OUTPUT_SIZE +
                               2.0 * INPUT_SIZE * HIDDEN_SIZE * BATCH_SIZE;
        double total_ops = ops_per_batch * train_batches * EPOCHS;
        double gflops = (total_ops / total_train_time) / 1e9;

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
    }

    // Free memory
    delete[] h_train_X; delete[] h_train_Y; delete[] h_test_X; delete[] h_test_Y;
    delete[] W1; delete[] W2;
    delete[] X; delete[] Y; delete[] H; delete[] O;
    delete[] dW1; delete[] dW2; delete[] dZ1; delete[] dZ2; delete[] dH;

    MPI_Finalize();
    return 0;
}
