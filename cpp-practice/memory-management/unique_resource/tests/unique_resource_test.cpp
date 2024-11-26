#include <gtest/gtest.h>
#include "unique_resource/unique_resource.h"

struct TestStruct
{
public:
    int value;
    explicit TestStruct(int v) : value(v) {}
};

// Basic Construction/Destruction
TEST(UniqueResourceTest, DefaultConstruction)
{
    // Test default construction
    UniqueResource<TestStruct> uniq_ptr;

    ASSERT_EQ(nullptr, uniq_ptr.get());
}

TEST(UniqueResourceTest, ConstructWithPointer)
{
    // Test construction with a pointer
    TestStruct *p = new TestStruct(15);
    UniqueResource<TestStruct> uniq_ptr{p};

    ASSERT_EQ(p, uniq_ptr.get());
    ASSERT_EQ(15, uniq_ptr->value);
    ASSERT_EQ(15, (*uniq_ptr).value);
}

TEST(UniqueResourceTest, MoveConstruction)
{
    UniqueResource<TestStruct> original(new TestStruct(42));
    TestStruct *original_ptr = original.get();

    UniqueResource<TestStruct> moved(std::move(original));

    ASSERT_EQ(nullptr, original.get());   // Original should be null
    ASSERT_EQ(original_ptr, moved.get()); // Moved should have the pointer
    ASSERT_EQ(42, moved->value);          // Data should be accessible
}

TEST(UniqueResourceTest, MoveAssignment)
{
    // Test move assignment
    UniqueResource<TestStruct> original(new TestStruct(67));
    TestStruct *original_ptr = original.get();
    UniqueResource<TestStruct> moved = std::move(original);

    ASSERT_EQ(nullptr, original.get());   // Original should be null
    ASSERT_EQ(original_ptr, moved.get()); // Moved should have the pointer
    ASSERT_EQ(67, (*moved).value);        // Data should be accessible
}

// Resource Access
TEST(UniqueResourceTest, GetAccessor)
{
    // Test get() method
    TestStruct *p = new TestStruct(17);
    UniqueResource<TestStruct> uniq_ptr{p};

    ASSERT_EQ(p, uniq_ptr.get());
    ASSERT_EQ(17, (*uniq_ptr).value);
}

TEST(UniqueResourceTest, DereferenceOperator)
{
    // Test operator*
    TestStruct *p = new TestStruct(42);
    UniqueResource<TestStruct> uniq_ptr{p};

    ASSERT_EQ(42, (*uniq_ptr).value);
}

TEST(UniqueResourceTest, ArrowOperator)
{
    // Test operator->
    TestStruct *p = new TestStruct(5);
    UniqueResource<TestStruct> uniq_ptr{p};

    ASSERT_EQ(5, uniq_ptr->value);
}

// // Memory Management
// TEST(UniqueResourceTest, ResourceDeletion) {
//     // Test that resources are properly deleted
// }

// // Edge Cases
// TEST(UniqueResourceTest, NullPointerHandling) {
//     // Test handling of nullptr
// }