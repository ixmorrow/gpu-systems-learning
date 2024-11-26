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

    Matrix(size_t x, size_t y) : num_rows(x), num_cols(y) {};

    ~Matrix() {};

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

private:
    // matrix data stored in 1D array in row-major layout
    // convert 2D index to 1D index
    int _row_major_index(int row, int col) const
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