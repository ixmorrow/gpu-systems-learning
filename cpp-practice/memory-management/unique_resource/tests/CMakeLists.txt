add_executable(unique_resource_test unique_resource_test.cpp)
target_link_libraries(unique_resource_test GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unique_resource_test)