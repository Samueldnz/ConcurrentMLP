#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural_network.h"

#define ROWS 2000
#define COLS 6
#define INPUT_SIZE 5
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 1
#define EPOCHS 1000
#define LEARNING_RATE 0.1f
#define SAMPLES 2000

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

void train(float inputs[SAMPLES][INPUT_SIZE], float targets[SAMPLES]) {
    Neuron hidden[HIDDEN_SIZE];
    Neuron output;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        init_neuron_weights(&hidden[i], INPUT_SIZE, -1.0f, 1.0f);
    }
    init_neuron_weights(&output, HIDDEN_SIZE, -1.0f, 1.0f);

    clock_t start_time = clock();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_error = 0.0f;

        for (int s = 0; s < SAMPLES; s++) {
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE; j++)
                    hidden[i].x[j] = inputs[s][j];
                ReLU(&hidden[i]);
            }
            for (int i = 0; i < HIDDEN_SIZE; i++)
                output.x[i] = hidden[i].s;
            sigmoid(&output);

            float error = targets[s] - output.s;
            total_error += error * error;

            float d_output = error * derivative_sigmoid(output.s);

            float d_hidden[HIDDEN_SIZE + 1];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                d_hidden[i] = d_output * output.w[i] * derivative_ReLU(hidden[i].z);
            }

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    hidden[i].w[j] += LEARNING_RATE * d_hidden[i] * hidden[i].x[j];
                }
                hidden[i].b += LEARNING_RATE * d_hidden[i];
            }

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                output.w[i] += LEARNING_RATE * d_output * output.x[i];
            }
            output.b += LEARNING_RATE * d_output;
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
