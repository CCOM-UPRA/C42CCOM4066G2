from mpi4py import MPI
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[1])

rows = N // size
startRow = rank * rows
endRow = (rank + 1) * rows if rank != size - 1 else N

A = np.random.randint(1, 100, size=(N, N))
B = np.random.randint(1, 100, size=(N, N))

local_A = A[startRow:endRow, :]
local_B = B[startRow:endRow, :]

comm.Barrier()
startTime = time.time()

local_result = local_A * local_B

comm.Barrier()
endTime = time.time()

finished = comm.gather(local_result, root=0)

if rank == 0:
    full_result = np.vstack(finished)
    print(f"matriz {N}x{N} usando {size} procesos")
    print(f"Time: {endTime - startTime:.6f} segundos")
    print(full_result)
