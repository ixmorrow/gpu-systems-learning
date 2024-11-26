#pragma once

template <typename T>
class UniqueResource
{
private:
    // Add your member variables here
    T *ptr;

public:
    // Constructor
    explicit UniqueResource(T *obj = nullptr) : ptr(obj) {};

    // Destructor
    ~UniqueResource() noexcept
    {
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
        if (this != other)
        {
            // delete existing data on this object
            delete ptr;
            // assign ptr address from other to this object
            ptr = other.ptr;
            // set other ptr address to a default value of nullptr
            other.ptr = nullptr;
        }

        return *this;
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