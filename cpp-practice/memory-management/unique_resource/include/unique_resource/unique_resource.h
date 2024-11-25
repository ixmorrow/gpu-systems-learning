#pragma once

template <typename T>
class UniqueResource
{
private:
    // Add your member variables here

public:
    // Constructor
    explicit UniqueResource(/* What parameters might you need? */);

    // Destructor
    ~UniqueResource();

    // Copy constructor and assignment - think about what you should do with these
    UniqueResource(const UniqueResource &);
    UniqueResource &operator=(const UniqueResource &);

    // Move constructor and assignment
    UniqueResource(UniqueResource &&) noexcept;
    UniqueResource &operator=(UniqueResource &&) noexcept;

    // Resource access
    T *get() const;
    T &operator*() const;
    T *operator->() const;
};