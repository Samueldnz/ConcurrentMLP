#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include "neural_network.h"

#define ROWS 2000
#define COLS 6
#define INPUT_SIZE 5
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 1
#define EPOCHS 1000
#define LEARNING_RATE 0.1f
#define SAMPLES 2000
#define N_THREADS 4

typedef struct {
    int thread_id;
    float (*inputs)[INPUT_SIZE];
    float *targets;
    Neuron *hidden;
    Neuron *output;
    pthread_mutex_t *mutex;
    int start_sample;
    int end_sample;
    int start_hidden;
    int end_hidden;
    float *error_sum;
} ThreadData;

void read_csv_to_matrix(const char *filename, float matrix[ROWS][COLS]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Erro ao abrir o arquivo");
        exit(EXIT_FAILURE);
    }
    char line[1024];
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < ROWS) {
        char *ptr = line;
        for (int col = 0; col < COLS; col++) {
            matrix[row][col] = strtof(ptr, &ptr);
            if (*ptr == ',') ptr++;
        }
        row++;
    }
    fclose(file);
}

void* train_thread(void* arg) {
    ThreadData *data = (ThreadData*) arg;

    for (int s = data->start_sample; s < data->end_sample; s++) {
        for (int i = data->start_hidden; i < data->end_hidden; i++) {
            for (int j = 0; j < INPUT_SIZE; j++)
                data->hidden[i].x[j] = data->inputs[s][j];
            ReLU(&data->hidden[i]);
        }

        if (data->thread_id == 0) {
            for (int i = 0; i < HIDDEN_SIZE; i++)
                data->output->x[i] = data->hidden[i].s;
            sigmoid(data->output);

            float error = data->targets[s] - data->output->s;
            float d_output = error * derivative_sigmoid(data->output->s);
            float d_hidden[HIDDEN_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++)
                d_hidden[i] = d_output * data->output->w[i] * derivative_ReLU(data->hidden[i].z);

            pthread_mutex_lock(data->mutex);
            *data->error_sum += error * error;
            pthread_mutex_unlock(data->mutex);

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    data->hidden[i].w[j] += LEARNING_RATE * d_hidden[i] * data->hidden[i].x[j];
                }
                data->hidden[i].b += LEARNING_RATE * d_hidden[i];
            }

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                data->output->w[i] += LEARNING_RATE * d_output * data->output->x[i];
            }
            data->output->b += LEARNING_RATE * d_output;
        }
    }

    return NULL;
}

void train(float inputs[SAMPLES][INPUT_SIZE], float targets[SAMPLES]) {
    Neuron hidden[HIDDEN_SIZE];
    Neuron output;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        init_neuron_weights(&hidden[i], INPUT_SIZE, -1.0f, 1.0f);
    }
    init_neuron_weights(&output, HIDDEN_SIZE, -1.0f, 1.0f);

    clock_t start_time = clock();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_error = 0.0f;
        pthread_t threads[N_THREADS];
        ThreadData thread_data[N_THREADS];

        for (int t = 0; t < N_THREADS; t++) {
            thread_data[t].thread_id = t;
            thread_data[t].inputs = inputs;
            thread_data[t].targets = targets;
            thread_data[t].hidden = hidden;
            thread_data[t].output = &output;
            thread_data[t].mutex = &mutex;
            thread_data[t].start_sample = 0;
            thread_data[t].end_sample = SAMPLES;
            thread_data[t].start_hidden = (HIDDEN_SIZE / N_THREADS) * t;
            thread_data[t].end_hidden = (t == N_THREADS - 1) ? HIDDEN_SIZE : thread_data[t].start_hidden + (HIDDEN_SIZE / N_THREADS);
            thread_data[t].error_sum = &total_error;
            pthread_create(&threads[t], NULL, train_thread, &thread_data[t]);
        }

        for (int t = 0; t < N_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }

        printf("Época %d - Erro médio quadrático: %.6f\n", epoch, total_error / SAMPLES);
    }

    clock_t end_time = clock();
    float time_taken = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("\n⏱️ Tempo total de treinamento: %.2f segundos\n", time_taken);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        free_neuron(&hidden[i]);
    }
    free_neuron(&output);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <arquivo.csv>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    float data[ROWS][COLS];
    float inputs[SAMPLES][INPUT_SIZE];
    float targets[SAMPLES];

    read_csv_to_matrix(argv[1], data);

    for (int i = 0; i < SAMPLES; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            inputs[i][j] = data[i][j];
        }
        targets[i] = data[i][INPUT_SIZE];
    }

    train(inputs, targets);

    return 0;
}
