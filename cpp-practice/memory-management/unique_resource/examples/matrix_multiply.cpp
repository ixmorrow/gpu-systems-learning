#include "unique_resource/unique_resource.h"
#include <cstddef>
#include <stdexcept>

using namespace std;
class Matrix
{
public:
    size_t num_rows;
    size_t num_cols;
    UniqueResource<float> data;

    // Constructors and destructor
    Matrix() : num_rows(0), num_cols(0) {};
    Matrix(size_t x, size_t y) : num_rows(x), num_cols(y) {};
    Matrix(float *initial_data) : data(initial_data) {};

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

    void add(Matrix &other)
    // Adds other matrix to this matrix
    {
        if (num_rows != other.num_rows || num_cols != other.num_cols)
        {
            throw IncompatibleMatricesForOperation();
        }
        for (int i; i < num_rows; ++i)
        {
            for (int j; j < num_cols; ++j)
            {
                float a = get(i, j);
                float b = other.get(i, j);
                set(i, j, a + b);
            }
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