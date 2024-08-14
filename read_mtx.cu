#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

void read_matrix(const char* file_path, int*& row_ptr, int*& col_ind, double*& values, int& num_rows, int& num_cols) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    
    // Loop to skip comments and first line of info about the matrix
    bool skip = true;
    while (skip) {
        std::getline(file, line);
        if (line[0] == '%') continue; // Skip comments
        skip = false;
    }
    
    std::vector<int> row_indices, col_indices;
    std::vector<double> vals;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int row, col;
        double val = 1.0;  // Default value if only two values are provided
        if (!(iss >> row >> col)) break;  // Read row and column
        iss >> val;  // Try to read value if it exists
        row_indices.push_back(row - 1);
        col_indices.push_back(col - 1);
        vals.push_back(val);
    }

    num_rows = *std::max_element(row_indices.begin(), row_indices.end()) + 1;
    num_cols = *std::max_element(col_indices.begin(), col_indices.end()) + 1;
    int nnz = vals.size();

    row_ptr = new int[num_rows + 1]();
    col_ind = new int[nnz];
    values = new double[nnz];

    for (int i = 0; i < nnz; ++i) {
        row_ptr[row_indices[i] + 1]++;
    }

    for (int i = 1; i <= num_rows; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }

    std::vector<int> temp_row_ptr(num_rows + 1);
    std::copy(row_ptr, row_ptr + num_rows + 1, temp_row_ptr.begin());

    for (int i = 0; i < nnz; ++i) {
        int row = row_indices[i];
        int dest = temp_row_ptr[row];

        col_ind[dest] = col_indices[i];
        values[dest] = vals[i];

        temp_row_ptr[row]++;
    }

    file.close();
}