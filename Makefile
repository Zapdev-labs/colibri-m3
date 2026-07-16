CC ?= gcc
ARCH ?= native
CFLAGS = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-function -Isrc
LDFLAGS = -lm -fopenmp

.PHONY: all clean m3

all: m3

m3: src/engine.c src/st.h src/json.h src/planar_kv.h
	$(CC) $(CFLAGS) src/engine.c -o m3 $(LDFLAGS)

clean:
	rm -f m3
