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
    UniqueResource(UniqueResource &&) noexcept;
    UniqueResource &operator=(UniqueResource &&) noexcept;

    // Resource access
    T *get() const;
    T &operator*() const;
    T *operator->() const;
};