#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

// ==========================================
// 1. Hyperparameters & Network Architecture
// ==========================================
#define INPUT_SIZE 784   // 28x28 pixels
#define HIDDEN_SIZE 256  // Nodes in Hidden Layer
#define OUTPUT_SIZE 10   // Classes (Fashion-MNIST)
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
// 3. OpenMP Functions
// ==========================================
void matrixMul(float* A, float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float Cvalue = 0.0f;
            for (int p = 0; p < K; ++p) {
                Cvalue += A[row * K + p] * B[p * N + col];
            }
            C[row * N + col] = Cvalue;
        }
    }
}

void relu(float* d_out, int size) {
    #pragma omp parallel for
    for (int idx = 0; idx < size; ++idx) {
        d_out[idx] = fmaxf(0.0f, d_out[idx]);
    }
}

void softmax(float* d_out, int batch_size, int num_classes) {
    #pragma omp parallel for
    for (int row = 0; row < batch_size; ++row) {
        float max_val = d_out[row * num_classes];
        for (int i = 1; i < num_classes; ++i) {
            max_val = fmaxf(max_val, d_out[row * num_classes + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            d_out[row * num_classes + i] = expf(d_out[row * num_classes + i] - max_val);
            sum += d_out[row * num_classes + i];
        }
        for (int i = 0; i < num_classes; ++i) {
            d_out[row * num_classes + i] /= sum;
        }
    }
}

void evaluate_loss_acc(float* d_O, float* d_Y, float* d_loss, int* d_correct, int batch_size, int num_classes) {
    int local_correct = 0;
    float local_loss = 0.0f;

    #pragma omp parallel for reduction(+:local_correct, local_loss)
    for (int row = 0; row < batch_size; ++row) {
        int best_class = 0;
        float max_prob = d_O[row * num_classes];
        float prob_correct = 1e-7f; 

        for (int i = 0; i < num_classes; ++i) {
            float p = d_O[row * num_classes + i];
            if (p > max_prob) { 
                max_prob = p; 
                best_class = i; 
            }
            if (i == (int)d_Y[row]) prob_correct = fmaxf(p, 1e-7f); 
        }

        if (best_class == (int)d_Y[row]) local_correct++;
        local_loss += -logf(prob_correct); 
    }
    
    *d_correct += local_correct;
    *d_loss += local_loss;
}

void compute_dz2(float* d_O, float* d_Y, float* dz2, int batch_size, int num_classes) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < num_classes; ++col) {
            int idx = row * num_classes + col;
            float y_val = (d_Y[row] == col) ? 1.0f : 0.0f;
            dz2[idx] = (d_O[idx] - y_val) / batch_size;
        }
    }
}

void matMulTransposeA(float* A, float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[i * M + row] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

void matMulTransposeB(float* A, float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[col * K + i];
            }
            C[row * N + col] = sum;
        }
    }
}

void relu_backward(float* dH, float* H, float* dZ1, int size) {
    #pragma omp parallel for
    for (int idx = 0; idx < size; ++idx) {
        dZ1[idx] = (H[idx] > 0.0f) ? dH[idx] : 0.0f;
    }
}

void update_weights(float* W, float* dW, float lr, int size) {
    #pragma omp parallel for
    for (int idx = 0; idx < size; ++idx) {
        W[idx] -= lr * dW[idx];
    }
}

// ==========================================
// 4. Main Training Loop
// ==========================================
int main() {
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

    float *d_X = new float[BATCH_SIZE * INPUT_SIZE];
    float *d_Y = new float[BATCH_SIZE];
    float *d_W1 = h_W1;
    float *d_W2 = h_W2;
    float *d_H = new float[BATCH_SIZE * HIDDEN_SIZE];
    float *d_O = new float[BATCH_SIZE * OUTPUT_SIZE];
    float *d_dW1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *d_dW2 = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    float *d_dZ1 = new float[BATCH_SIZE * HIDDEN_SIZE];
    float *d_dZ2 = new float[BATCH_SIZE * OUTPUT_SIZE];
    float *d_dH = new float[BATCH_SIZE * HIDDEN_SIZE];
    int d_correct;
    float d_loss;

    cout << "\nStarting Training..." << endl;

    double total_train_time = 0.0;
    float final_train_acc = 0.0f, final_val_acc = 0.0f, final_test_acc = 0.0f;
    float final_train_loss = 0.0f, final_val_loss = 0.0f, final_test_loss = 0.0f;

    // ==========================================
    // --- EPOCH LOOP (Train & Validation) ---
    // ==========================================
    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        auto train_start = chrono::high_resolution_clock::now();

        // --- 1. TRAIN PHASE ---
        int correct_train = 0;
        float train_loss_total = 0.0f;
        d_correct = 0;
        d_loss = 0.0f;

        for (int b = 0; b < train_batches; b++) {
            copy(h_train_X + b * BATCH_SIZE * INPUT_SIZE, h_train_X + (b + 1) * BATCH_SIZE * INPUT_SIZE, d_X);
            copy(h_train_Y + b * BATCH_SIZE, h_train_Y + (b + 1) * BATCH_SIZE, d_Y);

            matrixMul(d_X, d_W1, d_H, BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
            relu(d_H, BATCH_SIZE * HIDDEN_SIZE);

            matrixMul(d_H, d_W2, d_O, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
            softmax(d_O, BATCH_SIZE, OUTPUT_SIZE);

            evaluate_loss_acc(d_O, d_Y, &d_loss, &d_correct, BATCH_SIZE, OUTPUT_SIZE);

            compute_dz2(d_O, d_Y, d_dZ2, BATCH_SIZE, OUTPUT_SIZE);
            
            matMulTransposeA(d_H, d_dZ2, d_dW2, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
            matMulTransposeB(d_dZ2, d_W2, d_dH, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            
            relu_backward(d_dH, d_H, d_dZ1, BATCH_SIZE * HIDDEN_SIZE);
            matMulTransposeA(d_X, d_dZ1, d_dW1, INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE);

            update_weights(d_W1, d_dW1, LEARNING_RATE, INPUT_SIZE * HIDDEN_SIZE);
            update_weights(d_W2, d_dW2, LEARNING_RATE, HIDDEN_SIZE * OUTPUT_SIZE);
        }

        auto train_end = chrono::high_resolution_clock::now();
        chrono::duration<double> epoch_time = train_end - train_start;
        total_train_time += epoch_time.count();

        correct_train = d_correct;
        train_loss_total = d_loss;
        final_train_acc = (float)correct_train / (train_batches * BATCH_SIZE) * 100.0f;
        final_train_loss = train_loss_total / (train_batches * BATCH_SIZE);

        // --- 2. VALIDATION PHASE ---
        int correct_val = 0;
        float val_loss_total = 0.0f;
        d_correct = 0;
        d_loss = 0.0f;

        for (int b = 0; b < val_batches; b++) {
            copy(h_train_X + (train_samples + b * BATCH_SIZE) * INPUT_SIZE, h_train_X + (train_samples + (b + 1) * BATCH_SIZE) * INPUT_SIZE, d_X);
            copy(h_train_Y + train_samples + b * BATCH_SIZE, h_train_Y + train_samples + (b + 1) * BATCH_SIZE, d_Y);

            matrixMul(d_X, d_W1, d_H, BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
            relu(d_H, BATCH_SIZE * HIDDEN_SIZE);

            matrixMul(d_H, d_W2, d_O, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
            softmax(d_O, BATCH_SIZE, OUTPUT_SIZE);

            evaluate_loss_acc(d_O, d_Y, &d_loss, &d_correct, BATCH_SIZE, OUTPUT_SIZE);
        }
        
        correct_val = d_correct;
        val_loss_total = d_loss;
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
    d_correct = 0;
    d_loss = 0.0f;

    for (int b = 0; b < test_batches; b++) {
        copy(h_test_X + b * BATCH_SIZE * INPUT_SIZE, h_test_X + (b + 1) * BATCH_SIZE * INPUT_SIZE, d_X);
        copy(h_test_Y + b * BATCH_SIZE, h_test_Y + (b + 1) * BATCH_SIZE, d_Y);

        matrixMul(d_X, d_W1, d_H, BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
        relu(d_H, BATCH_SIZE * HIDDEN_SIZE);

        matrixMul(d_H, d_W2, d_O, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
        softmax(d_O, BATCH_SIZE, OUTPUT_SIZE);

        evaluate_loss_acc(d_O, d_Y, &d_loss, &d_correct, BATCH_SIZE, OUTPUT_SIZE);
    }
    
    correct_test = d_correct;
    test_loss_total = d_loss;
    final_test_acc = (float)correct_test / (test_batches * BATCH_SIZE) * 100.0f;
    final_test_loss = test_loss_total / (test_batches * BATCH_SIZE);

    // --- GFLOPS Calculation ---
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

    delete[] d_X; delete[] d_Y; delete[] d_H; delete[] d_O;
    delete[] d_dW1; delete[] d_dW2; delete[] d_dZ1; delete[] d_dZ2; delete[] d_dH;
    delete[] h_train_X; delete[] h_train_Y; delete[] h_test_X; delete[] h_test_Y;
    delete[] h_W1; delete[] h_W2;

    return 0;
}
