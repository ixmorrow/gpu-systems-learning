# Mojo - SIMD Type

I am about halfway through my second tutorial with Mojo. The first tutorial was a fairly basic one and did not incorporate a lot of what makes Mojo so unique. However, in the second tutorial, we are writing an optimized matrix multiplication class in Mojo and comparing it to the performance of a plain Python matrix multiplication. During this tutorial, I learned about the `SIMD` type in Mojo. SIMD stands for Single Instruction, Multiple Data and it’s a core concept for parallel programming in the ML space. It’s a very important topic in CUDA programming as well. The main idea of SIMD is that you can perform the same instruction over different pieces of data in parallel to speed up execution time. In CUDA, this is done by passing multiple instructions with different parts of a collection of data off to an Nvidia GPU’s Streaming Multiprocessors, of which there are many depending on which Nvidia GPU you are using (64+). 

A SIMD object is a vector. When defining a SIMD object, you must provide two parameters:

- a `DType` value, defining the data type in the vector
- number of elements in the vector, which must be a power of two
- `x = SIMD[DType.uint8, 4](1, 2, 3, 4)`
- here we define a SIMD vector to hold 4 unsigned 8-bit ints at a time and it’s initialized with the values 1, 2, 3, 4

SIMD allows a single instruction to be executed across the multiple data elements of the vector. SIMD represents a small vector that is backed by a hardware vector element.

SIMD reminds me of a map in TensorFlow. With SIMD vectors, we can operate on a number of elements in parallel without the GPU. Modern CPUs have SIMD specific registers whose purpose is to operator on data in parallel with SIMD, they have the ability to operate a single instruction on all data that is stored in the register.

Interesting enough, Mojo has a few default alias types that are just SIMD vectors under the hood.

- `Scalar` → SIMD vec with a single element

This is the first language besides CUDA that I have seen where SIMD operations are treated as a first class citizen. I think it really shows what the Modular team is thinking and what their goals are for the Mojo language. I will post more as I work my way through this and some other Mojo tutorials. I also plan to benchmark Mojo's matrix multiplication against vanilla Python, PyTorch, and CUDA!

[Mojo MatMul Tutorial](https://docs.modular.com/mojo/notebooks/Matmul/)