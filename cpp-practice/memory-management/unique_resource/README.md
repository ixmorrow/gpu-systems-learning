# UniqueResource Smart Pointer Implementation

## Overview
A custom implementation of a unique pointer-like resource manager to demonstrate RAII and modern C++ memory management principles.

## Features
- RAII resource management
- Move semantics
- Custom deleters
- Memory leak detection

## Building
```bash
mkdir build
cd build
cmake ..
make
```

## UniqueResource Implementation Project Outline

Requirements:
1. Create a template class that manages a single resource
2. Implement RAII principles
3. Prevent resource copying
4. Allow resource moving
5. Provide pointer-like access to the resource

Key concepts to consider:
- Constructor and destructor behavior 
- Copy/move semantics
- Resource access patterns
- Exception safety

Suggested steps:
1. Design the public interface
2. Implement basic resource management
3. Add move operations
4. Add access operators
5. Write test cases
