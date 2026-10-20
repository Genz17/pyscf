
gcc -O3 -std=c99 -fPIC -shared -fopenmp \
    -I.. -I../deps/include \
    pari.c -o ../libpari.so \
    -L.. -L../deps/lib \
    -Wl,-rpath,'$ORIGIN:$ORIGIN/deps/lib' \
    -Wl,-z,defs \
    -lcgto -lcint -lopenblas -lm

#!/usr/bin/env bash

export OMP_NUM_THREADS=16
export OMP_DYNAMIC=FALSE


set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
omp_prefix="$(brew --prefix libomp)"

clang -O3 -std=c99 -fPIC -dynamiclib \
    -Xpreprocessor -fopenmp \
    -I.. -I../deps/include -I"$omp_prefix/include" \
    pari.c -o ../libpari.dylib \
    -L.. -L../deps/lib -L"$omp_prefix/lib" \
    -Wl,-rpath,@loader_path \
    -Wl,-rpath,@loader_path/deps/lib \
    "-Wl,-rpath,$omp_prefix/lib" \
    -lcgto -lcint -lomp -framework Accelerate
