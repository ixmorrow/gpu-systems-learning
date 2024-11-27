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

    static Matrix build_matrix(size_t rows, size_t cols)
    {
        Matrix m;
        m.num_rows = rows;
        m.num_cols = cols;
        m.data = UniqueResource<float>(new float[rows * cols]);

        return m;
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
            return data[i];
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
            data[index] = i;
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

        Matrix result = build_matrix(num_rows, num_cols);

        for (int i = 0; i < num_rows; ++i)
        {
            for (int j = 0; j < num_cols; ++j)
            {
                size_t global_index = _row_major_index(i, j);
                result.data[global_index] = data[global_index] + other.data[global_index];
            }
        }

        return result;
    }

    Matrix operator*(Matrix const &other)
    {
        if (num_cols != other.num_rows)
        {
            throw IncompatibleMatricesForOperation();
        }

        // create temporary matrix data
        Matrix result = build_matrix(num_rows, other.num_cols);

        for (int i = 0; i < num_rows; ++i) // Iterate over rows of first matrix
        {
            for (int j = 0; j < other.num_cols; ++j) // Iterates over cols of second matrix
            {
                float value = 0.0;
                for (int k = 0; k < num_cols; ++k) // Iterates over cols of first matrix
                // Computes dot product of ith row of 1st matrix and jth col of 2nd matrix
                {
                    value += data[i * num_cols + k] * other.data[k * other.num_cols + j];
                }
                result.data[i * num_cols + j] = value;
            }
        }

        return result;
    }

    void T()
    {
        // create temporary matrix data
        UniqueResource<float> temp{new float[num_cols * num_rows], true};

        for (int i = 0; i < num_rows; ++i)
        {
            for (int j = 0; j < num_cols; ++j)
            {
                size_t new_row = j;
                size_t new_col = i;
                size_t original_index = _row_major_index(i, j);
                size_t new_index = _row_major_index(new_row, new_col);

                temp[new_index] = data[original_index];
            }
        }

        // move ownership of temp resources to data
        // data now holds Transposed matrix!
        data = std::move(temp);
        std::swap(num_rows, num_cols);
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

    cout << "-------------------------------" << endl;

    // Transpose matrix C
    C.T();
    cout << "Transpose of Matrix C:" << endl;
    C.print();

    cout << "-------------------------------" << endl;
    cout << "Matrix Multiplication" << endl;
    float *array_d = new float[4]{2.0, -2.0, 5.0, 3.0};
    float *array_e = new float[4]{-1.0, 4.0, 7.0, -6.0};
    Matrix D{2, 2, array_d};
    Matrix E{2, 2, array_e};

    cout << "Matrix D:" << endl;
    D.print();
    cout << "Matrix E:" << endl;
    E.print();

    Matrix F = D * E;
    cout << "Result of D*E:" << endl;
    F.print();

    return 0;
}