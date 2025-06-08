#include <stdio.h>
#include <stdlib.h>
#include "data.h"

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
            if (*ptr == ',') ptr++;
        }
        row++;
    }

    fclose(file);
}

void extract_inputs_targets(double data[ROWS][COLS], double inputs[SAMPLES][INPUT_SIZE], double targets[SAMPLES]) {
    for (int i = 0; i < SAMPLES; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            inputs[i][j] = data[i][j];
        }
        targets[i] = data[i][INPUT_SIZE];
    }
}
