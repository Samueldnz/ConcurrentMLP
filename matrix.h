// 
float** read_csv (const char* file_name);

// 
void free_matrix (float** matrix, int rows);

// 
float*** create_k_folds (float** matrix, int rows, int columns, int k);