#include <gtest/gtest.h>
#include "unique_resource/unique_resource.h"

struct TestStruct
{
public:
    int value;
    explicit TestStruct(int v) : value(v) {}
};

class MemoryTestClass
{
public:
    static int constructor_calls;
    static int destructor_calls;

    explicit MemoryTestClass(int val = 0) : value(val)
    {
        constructor_calls++;
    }

    ~MemoryTestClass()
    {
        destructor_calls++;
    }

    int value;
};

// Reset counters before each test
class UniqueResourceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MemoryTestClass::constructor_calls = 0;
        MemoryTestClass::destructor_calls = 0;
    }
};

// Initialize static counters
int MemoryTestClass::constructor_calls = 0;
int MemoryTestClass::destructor_calls = 0;

// Basic Construction/Destruction
TEST_F(UniqueResourceTest, DefaultConstruction)
{
    // Test default construction
    UniqueResource<TestStruct> uniq_ptr;

    ASSERT_EQ(nullptr, uniq_ptr.get());
}

TEST_F(UniqueResourceTest, ConstructWithPointer)
{
    // Test construction with a pointer
    TestStruct *p = new TestStruct(15);
    UniqueResource<TestStruct> uniq_ptr{p};

    ASSERT_EQ(p, uniq_ptr.get());
    ASSERT_EQ(15, uniq_ptr->value);
    ASSERT_EQ(15, (*uniq_ptr).value);
}

TEST_F(UniqueResourceTest, MoveConstruction)
{
    UniqueResource<TestStruct> original(new TestStruct(42));
    TestStruct *original_ptr = original.get();

    UniqueResource<TestStruct> moved(std::move(original));

    ASSERT_EQ(nullptr, original.get());   // Original should be null
    ASSERT_EQ(original_ptr, moved.get()); // Moved should have the pointer
    ASSERT_EQ(42, moved->value);          // Data should be accessible
}

TEST_F(UniqueResourceTest, MoveAssignment)
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
TEST_F(UniqueResourceTest, GetAccessor)
{
    // Test get() method
    TestStruct *p = new TestStruct(17);
    UniqueResource<TestStruct> uniq_ptr{p};

    ASSERT_EQ(p, uniq_ptr.get());
    ASSERT_EQ(17, (*uniq_ptr).value);
}

TEST_F(UniqueResourceTest, DereferenceOperator)
{
    // Test operator*
    TestStruct *p = new TestStruct(42);
    UniqueResource<TestStruct> uniq_ptr{p};

    ASSERT_EQ(42, (*uniq_ptr).value);
}

TEST_F(UniqueResourceTest, ArrowOperator)
{
    // Test operator->
    TestStruct *p = new TestStruct(5);
    UniqueResource<TestStruct> uniq_ptr{p};

    ASSERT_EQ(5, uniq_ptr->value);
}

// Memory Management
TEST_F(UniqueResourceTest, ResourceDeletion)
{
    // Test that resources are properly deleted
    {
        UniqueResource<MemoryTestClass> uniq_ptr(new MemoryTestClass(75));
        EXPECT_EQ(MemoryTestClass::constructor_calls, 1);
        EXPECT_EQ(MemoryTestClass::destructor_calls, 0);
    } // ptr goes out of scope here
    EXPECT_EQ(MemoryTestClass::destructor_calls, 1);
}

TEST_F(UniqueResourceTest, MoveConstructionMemoryManagment)
{
    {
        UniqueResource<MemoryTestClass> original(new MemoryTestClass(75));
        EXPECT_EQ(MemoryTestClass::constructor_calls, 1);
        EXPECT_EQ(MemoryTestClass::destructor_calls, 0);
        {
            UniqueResource<MemoryTestClass> moved(std::move(original));
            EXPECT_EQ(MemoryTestClass::constructor_calls, 1);
        } // moved goes out of scope, destructor called
        EXPECT_EQ(MemoryTestClass::destructor_calls, 1);
    } // original goes out of scope, but destructor is not called
    EXPECT_EQ(MemoryTestClass::destructor_calls, 1);
}