#include "unique_resource/unique_resource.h"
#include <cstddef>
#include <stdexcept>
#include <random>
#include <iostream>

using namespace std;
class NullPtrMatrix : public exception
{
public:
    const char *what() const throw()
    {
        return "Retrieved a matrix that is a nullptr!";
    }
};

class IncompatibleMatricesForOperation : public exception
{
public:
    const char *what() const throw()
    {
        return "Attempted a matrix operatioin with two matrices with incompatble dimensions.";
    }
};

class Matrix
{
public:
    size_t num_rows;
    size_t num_cols;
    UniqueResource<float> data;

    // Constructors and destructor
    Matrix() : num_rows(0), num_cols(0), data(nullptr) {};
    Matrix(size_t x, size_t y) : num_rows(x), num_cols(y), data(new float[x * y], true) {};
    Matrix(size_t x, size_t y, float *initial_data)
        : num_rows(x), num_cols(y), data(initial_data, true) {};

    ~Matrix() {};

    // Move constructor
    Matrix(Matrix &&other) noexcept : num_rows(other.num_rows), num_cols(other.num_cols)
    {
        data = std::move(other.data);
    }

    // Move assignment
    Matrix &operator=(Matrix &&other) noexcept
    {
        // check not assigning to self
        if (this != &other)
        {
            // don't need to delete data b/c move assignment
            // will delete the original ptr
            data = std::move(other.data);
            num_rows = other.num_rows;
            num_cols = other.num_cols;
            other.num_rows = 0;
            other.num_cols = 0;
        }

        return *this;
    }

    // getter function to retrieve data given x and y coordinates in 2D array
    float get(size_t x, size_t y) const
    {
        if (x < num_rows && x >= 0 && y < num_cols && y >= 0 && _validate_matrix())
        {
            int i = _row_major_index(x, y);
            auto *matrix = data.get();
            return matrix[i];
        }
        else
        {
            throw std::out_of_range("Index out of range.");
        }
    }

    void set(size_t x, size_t y, float i)
    {
        if (x < num_rows && x >= 0 && y < num_cols && y >= 0 && _validate_matrix())
        {
            size_t index = _row_major_index(x, y);
            auto *matrix = data.get();
            matrix[index] = i;
        }
        else
        {
            throw std::out_of_range("Index out of range.");
        }
    }

    size_t get_num_rows() { return num_rows; }

    size_t get_num_cols() { return num_cols; }

    bool is_empty() { return data.get() == nullptr; }

    void reset()
    {
        num_rows = 0;
        num_cols = 0;
        UniqueResource<float> temp;
        data = std::move(temp);
    }

    Matrix operator+(Matrix const &other)
    {
        if (num_rows != other.num_rows || num_cols != other.num_cols)
        {
            throw IncompatibleMatricesForOperation();
        }

        Matrix result(num_rows, num_cols, new float[num_rows * num_cols]);

        for (int i = 0; i < num_rows; ++i)
        {
            for (int j = 0; j < num_cols; ++j)
            {
                size_t global_index = _row_major_index(i, j);
                auto *matrix = data.get();
                auto *other_matrix = other.data.get();
                auto *result_matrix = result.data.get();
                result_matrix[global_index] = matrix[global_index] + other_matrix[global_index];
            }
        }

        return result;
    }

    void print()
    {
        for (int i = 0; i < num_rows; ++i)
        {
            for (int j = 0; j < num_cols; ++j)
            {
                cout << get(i, j) << " ";
            }
            cout << endl;
        }
    }

private:
    // matrix data stored in 1D array in row-major layout
    // convert 2D index to 1D index
    size_t _row_major_index(size_t row, size_t col) const
    {
        return (row * num_cols) + col;
    }

    // validate matrix stored in data ptr is not null
    bool _validate_matrix() const
    {
        if (data.get() == nullptr)
        {
            throw NullPtrMatrix();
        }
        return true;
    }
};

int main()
{
    // create 10x10 matrix
    size_t x = 10;
    size_t y = 10;
    float *array_a = new float[x * y];
    float *array_b = new float[x * y];

    // Create a random number generator
    std::random_device rd;                               // Seed for randomness
    std::mt19937 gen(rd());                              // Mersenne Twister engine
    std::uniform_real_distribution<float> dis(0.0, 1.0); // Range [0.0, 1.0]

    for (int i = 0; i < (x * y); ++i)
    {
        array_a[i] = dis(gen);
        array_b[i] = dis(gen);
    }
    cout << "Initialized random arrays of size: " << x * y << endl;

    // create matrix A
    Matrix A{x, y, array_a};
    cout << "Matrix A: " << endl;
    A.print();

    cout << "-------------------------------" << endl;

    // create matrix B
    Matrix B{x, y, array_b};
    cout << "Matrix B: " << endl;
    B.print();
    cout << "-------------------------------" << endl;

    // test Matrix addition
    Matrix C = A + B;
    cout << "Result of Matrix Addition (A+B)" << endl;
    C.print();

    return 0;
}