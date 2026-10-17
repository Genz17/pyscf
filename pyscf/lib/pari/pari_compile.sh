gcc -O3 -std=c99 -fPIC -shared -fopenmp \
    -I.. -I../deps/include \
    pari.c -o ../libpari.so \
    -L.. -L../deps/lib \
    -Wl,-rpath,'$ORIGIN:$ORIGIN/deps/lib' \
    -Wl,-z,defs \
    -lcgto -lcint -lopenblas
