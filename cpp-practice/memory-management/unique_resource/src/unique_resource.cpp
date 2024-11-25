#include "unique_resource.h"

template <typename T>
class UniqueResource
{
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