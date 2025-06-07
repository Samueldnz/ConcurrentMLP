#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ROWS 2000
#define COLS 6
#define INPUT_SIZE 5
#define HIDDEN_SIZE 10
#define OUTPUT_SIZE 1
#define EPOCHS 1000
#define LEARNING_RATE 0.1
#define SAMPLES 2000 // quantidade de amostras

void read_csv_to_matrix(const char *filename, double matrix[ROWS][COLS]) {
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
            matrix[row][col] = strtod(ptr, &ptr);
            if (*ptr == ',') ptr++;  // Avança o ponteiro após a vírgula
        }
        row++;
    }

    fclose(file);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

void init_weights(float *w, int size, float min, float max) {
    for (int i = 0; i < size; i++) {
        w[i] = ((float) rand() / RAND_MAX) * (max - min) + min;
    }
}

void train(float inputs[SAMPLES][INPUT_SIZE], float targets[SAMPLES]) {
    float w1[HIDDEN_SIZE][INPUT_SIZE];
    float b1[HIDDEN_SIZE];
    float w2[OUTPUT_SIZE][HIDDEN_SIZE];
    float b2[OUTPUT_SIZE];

    // Inicialização
    init_weights(&w1[0][0], HIDDEN_SIZE * INPUT_SIZE, -1.0f, 1.0f);
    init_weights(b1, HIDDEN_SIZE, -1.0f, 1.0f);
    init_weights(&w2[0][0], OUTPUT_SIZE * HIDDEN_SIZE, -1.0f, 1.0f);
    init_weights(b2, OUTPUT_SIZE, -1.0f, 1.0f);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_error = 0;

        for (int s = 0; s < SAMPLES; s++) {
            float hidden[HIDDEN_SIZE];
            float output[OUTPUT_SIZE];

            // FORWARD
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

            float error = targets[s] - output[0];
            total_error += error * error;

            // BACKPROPAGATION
            float d_output = error * sigmoid_derivative(output[0]);

            float d_hidden[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                d_hidden[i] = d_output * w2[0][i] * sigmoid_derivative(hidden[i]);
            }

            // Atualizar pesos w2 e bias b2
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                w2[0][i] += LEARNING_RATE * d_output * hidden[i];
            }
            b2[0] += LEARNING_RATE * d_output;

            // Atualizar pesos w1 e bias b1
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    w1[i][j] += LEARNING_RATE * d_hidden[i] * inputs[s][j];
                }
                b1[i] += LEARNING_RATE * d_hidden[i];
            }
        }

        if (epoch % 100 == 0) {
            printf("Época %d - Erro médio quadrático: %.4f\n", epoch, total_error / SAMPLES);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <arquivo.csv>\n", argv[0]);
        return 1;
    }

    double data[ROWS][COLS];
    float inputs[SAMPLES][INPUT_SIZE];
    float targets[SAMPLES];

    read_csv_to_matrix(argv[1], data);

    // Transfere os dados para a rede
    for (int i = 0; i < SAMPLES; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            inputs[i][j] = (float)data[i][j]; // entrada
        }
        targets[i] = (float)data[i][INPUT_SIZE]; // última coluna como saída
    }

    train(inputs, targets);

    return 0;int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <arquivo.csv>\n", argv[0]);
        return 1;
    }

    double data[ROWS][COLS];
    read_csv_to_matrix(argv[1], data);

    float inputs[SAMPLES][INPUT_SIZE];
    float targets[SAMPLES];

    train(inputs, targets);
    
    return 0;
}

}
