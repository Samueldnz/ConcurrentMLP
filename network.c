#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "network.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double s) {
    return s * (1 - s);
}

void init_weights(double *w, int size, double min, double max) {
    for (int i = 0; i < size; i++) {
        w[i] = ((double) rand() / RAND_MAX) * (max - min) + min;
    }
}

void train(double inputs[SAMPLES][INPUT_SIZE], double targets[SAMPLES]) {
    double w1[HIDDEN_SIZE][INPUT_SIZE];
    double b1[HIDDEN_SIZE];
    double w2[OUTPUT_SIZE][HIDDEN_SIZE];
    double b2[OUTPUT_SIZE];

    init_weights(&w1[0][0], HIDDEN_SIZE * INPUT_SIZE, -1.0, 1.0);
    init_weights(b1, HIDDEN_SIZE, -1.0, 1.0);
    init_weights(&w2[0][0], OUTPUT_SIZE * HIDDEN_SIZE, -1.0, 1.0);
    init_weights(b2, OUTPUT_SIZE, -1.0, 1.0);

    clock_t start_time = clock();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0.0;

        for (int s = 0; s < SAMPLES; s++) {
            double hidden[HIDDEN_SIZE];
            double output[OUTPUT_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                hidden[i] = b1[i];
                for (int j = 0; j < INPUT_SIZE; j++) {
                    hidden[i] += w1[i][j] * inputs[s][j];
                }
                hidden[i] = sigmoid(hidden[i]);
            }

            for (int i = 0; i < OUTPUT_SIZE; i++) {
                output[i] = b2[i];
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    output[i] += w2[i][j] * hidden[j];
                }
                output[i] = sigmoid(output[i]);
            }

            double error = targets[s] - output[0];
            total_error += error * error;

            double d_output = error * sigmoid_derivative(output[0]);

            double d_hidden[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                d_hidden[i] = d_output * w2[0][i] * sigmoid_derivative(hidden[i]);
            }

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                w2[0][i] += LEARNING_RATE * d_output * hidden[i];
            }
            b2[0] += LEARNING_RATE * d_output;

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    w1[i][j] += LEARNING_RATE * d_hidden[i] * inputs[s][j];
                }
                b1[i] += LEARNING_RATE * d_hidden[i];
            }
        }

        printf("Época %d - Erro médio quadrático: %.6f\n", epoch, total_error / SAMPLES);
    }

    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\n⏱️ Tempo total de treinamento: %.2f segundos\n", time_taken);
}
