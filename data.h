#ifndef DATA_H
#define DATA_H

#include "constants.h"

void read_csv_to_matrix(const char *filename, double matrix[ROWS][COLS]);
void extract_inputs_targets(double data[ROWS][COLS], double inputs[SAMPLES][INPUT_SIZE], double targets[SAMPLES]);

#endif
