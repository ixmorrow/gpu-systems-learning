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