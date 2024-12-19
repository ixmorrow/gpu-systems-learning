import benchmark
from memory import memset_zero, UnsafePointer
from random import rand, random_float64

alias type = DType.float32


struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[type]]

    # Initialize zeroeing all values
    fn __init__(out self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(out self, data: UnsafePointer[Scalar[type]]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(data.address, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store(y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int, //](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store(y * self.cols + x, val)


# Note that C, A, and B have types.
fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


alias M = 1024
alias N = 1024
alias K = 1024


@always_inline
fn bench[
    func: fn (Matrix, Matrix, Matrix) -> None
](base_gflops: Float64) raises:
    var C = Matrix[M, N]()
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    var secs = benchmark.run[test_fn](max_runtime_secs=1).mean()

    A.data.free()
    B.data.free()
    C.data.free()

    var gflops = ((2 * M * N * K) / secs) / 1e9
    var speedup: Float64 = gflops / base_gflops

    print(gflops, "GFLOP/s, a", speedup, "x speedup over Python")


def main():
    var python_gflops = 0.007199698143878505
    bench[matmul_naive](python_gflops)

    # print system CPU SIMD register size
    from sys.info import simdbitwidth, simdwidthof

    print(simdbitwidth())
