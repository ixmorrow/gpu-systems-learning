# GPU/ML Systems Engineering Learning Path

## End Goal
Develop skills and expertise to become a GPU/ML Systems Engineer, specifically targeting roles similar to the AI GPU Performance Engineer position at Modular. The role requires expertise in:
- GPU programming and optimization
- Deep understanding of computer architecture
- High-performance computing
- ML systems design and optimization

## Complete 6-Month Plan

### MONTH 1: C++ Foundations & CUDA Basics
1. C++ Review Focus:
   - Modern C++ (17/20) features
   - Memory management and RAII
   - Templates and generic programming
   - Smart pointers

2. CUDA Foundations:
   - Complete NVIDIA's CUDA Programming Course
   - Begin Georgia Tech's "High Performance Computing" course
   - Learn NSight Systems basics

3. Practical Project: High-Performance Matrix Operations
   - Implement in modern C++
   - Port to CUDA
   - Profile with NSight
   - Compare CPU vs GPU performance

### MONTH 2: Advanced CUDA & Computer Architecture
1. Learning Focus:
   - Complete "Programming Massively Parallel Processors"
   - Study memory hierarchies and caching
   - Deep dive into GPU architecture
   - Continue Georgia Tech course

2. Practical Project: Cache-Friendly Neural Network Inference
   - Implement in C++/CUDA
   - Focus on memory coalescing
   - Use shared memory optimization
   - Profile memory access patterns
   - Document performance improvements

### MONTH 3: ML Operations & GPU Optimization
1. Learning Focus:
   - Study PyTorch's C++ backend
   - Learn CUDA kernel optimization techniques
   - Understanding cuBLAS and cuDNN
   - Complete Georgia Tech course

2. Practical Project: Custom ML Operations
   - Implement common ML ops (softmax, layer norm)
   - Use C++ templates for type genericity
   - Optimize for different tensor shapes
   - Integrate with PyTorch's C++ API

### MONTH 4: Distributed Systems & ML Infrastructure
1. Learning Focus:
   - Multi-GPU programming
   - NCCL communication library
   - C++ networking (Boost.ASIO)
   - Distributed system design

2. Practical Project: Distributed ML Training System
   - Implement in C++/CUDA
   - Focus on efficient GPU communication
   - Handle distributed memory management
   - Implement gradient synchronization

### MONTH 5: ML Compilation & Advanced Optimization
1. Learning Focus:
   - Study ML compilation systems (IREE, TVM)
   - Advanced C++ template metaprogramming
   - GPU kernel fusion techniques
   - Memory optimization strategies

2. Practical Project: ML Compiler Optimization
   - Build expression template system for GPU ops
   - Implement kernel fusion optimization
   - Create compute graph optimizations
   - Profile and benchmark against existing solutions

### MONTH 6: Production ML Systems
1. Learning Focus:
   - ML serving architectures
   - Production system design
   - Error handling across CPU/GPU boundary
   - Performance monitoring and debugging

2. Practical Project: High-Performance Model Serving System
   - Build C++ inference server
   - Implement batching and dynamic shapes
   - Add comprehensive monitoring
   - Focus on production-ready error handling

## Weekly Schedule
Monday:
- 1hr: C++ study (Effective Modern C++)
- 2hr: CUDA/GPU coursework
- 2hr: Project implementation

Tuesday:
- 1hr: Computer architecture study
- 2hr: C++ coding challenges
- 2hr: Project implementation

Wednesday:
- 2hr: Georgia Tech course
- 2hr: CUDA programming practice
- 1hr: Read ML systems papers

Thursday:
- 3hr: Main project work
- 1hr: NSight profiling practice
- 1hr: Study open source codebases

Friday:
- 2hr: Project optimization
- 1hr: C++/CUDA integration practice
- 2hr: Code review and documentation

Weekend:
- Study ML infrastructure codebases
- Work on personal projects
- Review and plan next week

## Resources

### Essential Books
1. C++ Focus:
   - "A Tour of C++" by Bjarne Stroustrup
   - "Effective Modern C++" by Scott Meyers
   - "C++ Templates: The Complete Guide"
   - "C++ Concurrency in Action"

2. GPU/ML Focus:
   - "Programming Massively Parallel Processors"
   - "Computer Architecture: A Quantitative Approach"
   - "Deep Learning Systems" by Qi Huang

### Online Courses
1. NVIDIA's CUDA C++ Course
2. Georgia Tech's "High Performance Computing"
3. Coursera's "Parallel Programming in CUDA"

### Video Resources
1. CppCon talks on GPU programming
2. NVIDIA GTC sessions
3. C++ Weekly with Jason Turner

### Documentation
1. CUDA Programming Guide
2. NSight Systems User Guide
3. PyTorch C++ API docs
4. NVIDIA Developer Blog

### Practice Platforms
1. Compiler Explorer (godbolt.org)
2. Quick Bench
3. CUDA Samples Repository

### Open Source Codebases to Study
1. PyTorch C++ Frontend
2. TVM
3. IREE
4. ONNXRuntime

### Development Tools
1. NSight Systems
2. NSight Compute
3. CUDA Toolkit
4. Visual Studio or CLion with CUDA support
5. Valgrind and CUDA-memcheck

## Progress Tracking

### Completed
- [x] Repository setup
- [x] Learning plan development
- [x] C++ refresher on Exercism
- [x] Initial chapters of "A Tour of C++"
- [x] First C++ project: UniqueResource implementation

### In Progress
- [ ] Current project details
- [ ] Current learning focus

### Next Steps
- [ ] Next planned project
- [ ] Upcoming learning objectives

## Project Organization
Each project should follow the structure:

project_name/
├── CMakeLists.txt
├── include/
├── src/
├── tests/
├── examples/
└── README.md

## Notes
- This is a living document that should be updated as progress is made
- Each project should include performance benchmarks
- Document learning insights and challenges
- Keep track of useful resources discovered during learning

---
Last Updated: [Current Date]
