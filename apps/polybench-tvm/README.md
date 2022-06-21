# Polybench-TVM

Implementation of compatible polybench kernels within TVM.

| Benchmark    | Supported | Description                                    | Notes/Caveats         |
|--------------|-----------|------------------------------------------------|-----------------------|
| 2mm          | ✅        | 2 Matrix Multiplications (D=A.B; E=C.D)        |                       |
| 3mm          | ✅        | 3 Matrix Multiplications (E=A.B; F=C.D; G=E.F) |                       |
| adi          |           | Alternating Direction Implicit solver          |                       |
| atax         | ✅        | Matrix Transpose and Vector Multiplication     |                       |
| bicg         |           | BiCG Sub Kernel of BiCGStab Linear Solver      |                       |
| cholesky     |           | Cholesky Decomposition                         |                       |
| correlation  |           | Correlation Computation                        |                       |
| covariance   |           | Covariance Computation                         |                       |
| doitgen      |           | Multiresolution analysis kernel (MADNESS)      |                       |
| durbin       |           | Toeplitz system solver                         |                       |
| dynprog      |           | Dynamic programming (2D)                       |                       |
| fdtd-2d      |           | 2-D Finite Different Time Domain Kernel        |                       |
| fdtd-apml    |           | FDTD using Anisotropic Perfectly Matched Layer |                       |
| gauss-filter |           | Gaussian Filter                                |                       |
| gemm         | ✅        | Matrix-multiply C=alpha.A.B+beta.C             | Not computed in-place |
| gemver       | ⌛        | Vector Multiplication and Matrix Addition      |                       |
| gesummv      |           | Scalar, Vector and Matrix Multiplication       |                       |
| gramschmidt  |           | Gram-Schmidt decomposition                     |                       |
| jacobi-1D    |           | 1-D Jacobi stencil computation                 |                       |
| jacobi-2D    |           | 2-D Jacobi stencil computation                 |                       |
| lu           |           | LU decomposition                               |                       |
| ludcmp       |           | LU decomposition                               |                       |
| mvt          |           | Matrix Vector Product and Transpose            |                       |
| reg-detect   |           | 2-D Image processing                           |                       |
| seidel       |           | 2-D Seidel stencil computation                 |                       |
| symm         |           | Symmetric matrix-multiply                      |                       |
| syr2k        |           | Symmetric rank-2k operations                   |                       |
| syrk         |           | Symmetric rank-k operations                    |                       |
| trisolv      |           | Triangular solver                              |                       |
| trmm         |           | Triangular matrix-multiply                     |                       |
