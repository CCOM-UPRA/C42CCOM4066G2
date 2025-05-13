from mpi4py import MPI
import numpy as np
import sys

# Incializa el MPI
comm = MPI.COMM_WORLD
processRank = comm.Get_rank()    
processCount = comm.Get_size()    

# Recibe el tamaño de la matriz via el bash
matrixSize = int(sys.argv[1])

# Determina la cantidad de filas que cada proceso realizara
rowsPerProcess = [
    matrixSize // processCount + (1 if i < matrixSize % processCount else 0)
    for i in range(processCount)
]
# Determina cual sera la posicion incial de cada proceso
rowDisplacements = [sum(rowsPerProcess[:i]) for i in range(processCount)]

# Asigna la cantidad de filas al proceso 
localRowCount = rowsPerProcess[processRank]

# Guarda el espacio en memoria
localMatrix = np.empty((localRowCount, matrixSize), dtype=np.int32)
vectorX = np.empty(matrixSize,              dtype=np.int32)

# Inicializa la matriz entera y el vector, y divide la matriz en sectores que se enviaran a los demas procesos
if processRank == 0:
    fullMatrix = np.random.randint(1, 100, size=(matrixSize, matrixSize), dtype=np.int32)
    vectorX[:] = np.random.randint(1, 100, size=matrixSize, dtype=np.int32)
    for dest in range(1, processCount):
        start = rowDisplacements[dest]
        count = rowsPerProcess[dest]
        comm.Send(
            [fullMatrix[start:start+count, :], MPI.INT],
            dest=dest,
            tag=77
        )
        # Guarda el primer bloque de filas para el proceso 0
    localMatrix[:] = fullMatrix[:rowsPerProcess[0], :]
else:
    # Reparte la matriz a los demas procesos 
    comm.Recv(
        [localMatrix, MPI.INT],
        source=0,
        tag=77
    )

# Transmite el vector a todos los procesos comenzando en el proceso 0
comm.Bcast([vectorX, MPI.INT], root=0)

# Espera a todos los procesos y comienza el timer
comm.Barrier()
startTime = MPI.Wtime()

# Realiza la multiplicacion del vector
localResult = localMatrix.dot(vectorX)

# Adjunta todos los resultados y los introduce al vector en el proceso 0
processRowCounts = np.array(rowsPerProcess,    dtype='i')
processRowDisplacements = np.array(rowDisplacements, dtype='i')
if processRank == 0:
    resultVector = np.empty(matrixSize, dtype=np.int32)
else:
    resultVector = None

comm.Gatherv(
    sendbuf=[localResult, MPI.INT],
    recvbuf=[resultVector, (processRowCounts, processRowDisplacements), MPI.INT],
    root=0
)

# Espera todos los procesos y termina el timer
comm.Barrier()
endTime = MPI.Wtime()

# Encuentra el tiempo total de todos los procesos
localElapsed = endTime - startTime
maxElapsed = comm.reduce(localElapsed, op=MPI.MAX, root=0)

# Proceso numero 0 imprime el tiempo total de ejecucion
if processRank == 0:
    print(f"Computed y = Matrix[{matrixSize}×{matrixSize}] × Vector[{matrixSize}] in {maxElapsed:.6f} seconds")

