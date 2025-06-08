// 
float** malloc_matrix (int rows, int columns);

// 
void free_matrix (float** matrix, int rows);

// 
float** read_csv (const char* file_name, int columns, int* rows_out);

// 
float*** k_folds (float** matrix, int rows, int columns, int k);