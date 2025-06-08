#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "constants.h"
#include "data.h"
#include "network.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <arquivo.csv>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    double data[ROWS][COLS];
    double inputs[SAMPLES][INPUT_SIZE];
    double targets[SAMPLES];

    read_csv_to_matrix(argv[1], data);
    extract_inputs_targets(data, inputs, targets);

    train(inputs, targets);

    return 0;
}
