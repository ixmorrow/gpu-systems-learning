## RAII practice with custom Matrix class in C++

Spent some time over the holidays working on practicing RAII in C++ by implementing my own custom unique pointer class and incorporating that into a custom matrix class in C++. The goal of this project was to practice RAII, which stands for Resource Acquisition Is Initialization, work on my C++ skills, and work on a ML systems related project. This code does not use any CUDA currently, but that could change down the line with future iterations.

With this project, I created a custom unique pointer class called `UniqueResource`. This data struct will point to a matrice's data in the matrix class. The reason I chose to store the matrice's data in a pointer is because the matrices involved in machine learning can get very large. With these extremely large matrices, it would be inefficient to store it in a traditional member variable as opposed to a pointer that will simply point to a location in memory where the data is stored. With this, we can also enforce safety measures in the `UniqueResource` class.

In order to test the `UniqueResource` class, I decided to implement a custom `Matrix` class that utilizes the `UniqueResource`. The matrix class also provided an opportunity to work on OOP concepts in C++ and the following:

* Move semantics

* working with row major indexing

* function overloading

* Templating

### UniqueResource

Let's get started with the `UniqueResource` implementation. This is going to be a templated class with the following member variables: `T *ptr` and `bool is_array`. `ptr` is a pointer to any type T. `is_array` is a boolean flag that indicates whether or not `ptr` is an array.

The purpose of this class is to point to a unique object that nothing else can point to, so we do not want to allow copying of the pointer. To do this, we override the copy constructor and assignment operators.

```cpp
UniqueResource(const UniqueResource &) = delete;
UniqueResource &operator=(const UniqueResource &) = delete;
```

Next, let's define the move constructor. Teaching move semantics in their entirety are outside the scope of this article, but I would suggest googling and doing some reading online about them if you are not familiar. The general idea with move semantics is that we want to assign ownership of some data type from one object to another without actually copying the value of that data and storing it somewhere else in memory, which is what a traditional assignment does. With move semantics, we also ensure only a single object owns the data at one time. This accomplishes both our uniqueness goal and will come in handy when working with very large data structs that are inefficient to copy. So, our move constructor should assign the value of its `ptr` to the place in memory the `other.ptr` points to. It should also update the `other.ptr` value so that it no longer points to the object after the assignment.

```cpp
UniqueResource(UniqueResource &&other) noexcept : ptr(other.ptr)
{
    other.ptr = nullptr;
}
```

Next, we'll also need to implement a move assignment operator. The move assignment will be utilized when we want to move ownership from one object to another, whereas the move constructor is used a new object is being initialized using an rvalue. With the move assignment, we'll need to delete the existing data on the object being assigned to, move the ownership over, and then set the data on the "other" object to an initialied state. In our case, that means the following:
1. delete existing `ptr` on object being assigned to
2. assign value of `ptr` to value of `other.ptr`
3. nullify `other.ptr`

```cpp
    UniqueResource &operator=(UniqueResource &&other) noexcept
    {
        // check not assigning to self
        if (this != &other)
        {
            // delete existing data on this object
            if (is_array)
                delete[] ptr;
            else
                delete ptr;
            // assign ptr address from other to this object
            ptr = other.ptr;
            // set other ptr address to a default value of nullptr
            other.ptr = nullptr;
        }

        return *this;
    }
```

After this, I also added a few more overrides to make working with this data struct easier. For example, I added overrides for the following ops:
* equality
* inequality
* indexing
* resource access

Here is the final `UniqueResource` definition.

```cpp
#pragma once
#include <cstddef>
#include <stdexcept>

template <typename T>
class UniqueResource
{
private:
    T *ptr;
    bool is_array;

public:
    // Constructor
    explicit UniqueResource(T *obj = nullptr) : ptr(obj), is_array(false) {};
    // Constructor for arrays
    UniqueResource(T *obj, bool array_flag) : ptr(obj), is_array(array_flag) {}

    // Destructor
    ~UniqueResource() noexcept
    {
        if (is_array)
            delete[] ptr;
        else
            delete ptr;
    };

    // Copy constructor and assignment - think about what you should do with these
    // Want to disable copying in this class because it is meant to be Unique.
    // If we copied pointer addr from A to B and A destructor deleter ptr, that would B with a dangling ptr.
    UniqueResource(const UniqueResource &) = delete;
    UniqueResource &operator=(const UniqueResource &) = delete;

    // Move constructor and assignment
    UniqueResource(UniqueResource &&other) noexcept : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }

    UniqueResource &operator=(UniqueResource &&other) noexcept
    {
        // check not assigning to self
        if (this != &other)
        {
            // delete existing data on this object
            if (is_array)
                delete[] ptr;
            else
                delete ptr;
            // assign ptr address from other to this object
            ptr = other.ptr;
            // set other ptr address to a default value of nullptr
            other.ptr = nullptr;
        }

        return *this;
    }

    bool operator==(UniqueResource const &other) noexcept
    {
        return ptr == other.get();
    }

    bool operator!=(UniqueResource const &other) noexcept
    {
        return !(*this == other);
    }

    // Non-const version for modifiable access
    T &operator[](std::size_t index)
    {
        if (!ptr)
        {
            throw std::runtime_error("Not an array resource.");
        }
        return ptr[index];
    }

    // Const version for read-only access
    const T &operator[](std::size_t index) const
    {
        if (!ptr)
        {
            throw std::runtime_error("Null pointer access");
        }
        return ptr[index];
    }

    // Resource access
    T *get() const
    {
        return ptr;
    }

    T &operator*() const
    {
        return *ptr;
    }

    T *operator->() const
    {
        return ptr;
    }
};
```

### Matrix Multiply Example with UniqueResource

To showcase how the `UniqueResource` class can be utilized in the context of machine learning, I implemented a `Matrix` class that uses the `UniqueResource` data struct to store matrices and perform operations on them. This implementation does not utilize CUDA at all, but it could in future iterations. Here is the full code for the example.

```cpp
#include "unique_resource/unique_resource.h"
#include <cstddef>
#include <stdexcept>
#include <random>
#include <iostream>
#include <immintrin.h>

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
```

