CC = mpicxx

CFLAGS += -fsycl -DMKL_ILP64 -fiopenmp
LIBS = -qmkl=parallel

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

all: gemm

gemm: gemm.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm *.o gemm
