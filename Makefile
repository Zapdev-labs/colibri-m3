CC ?= gcc
ARCH ?= native
CFLAGS = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-function -Isrc
LDFLAGS = -lm -fopenmp

.PHONY: all clean m3 test test-c test-converter check

all: m3

m3: src/engine.c src/st.h src/json.h src/planar_kv.h
	$(CC) $(CFLAGS) src/engine.c -o m3 $(LDFLAGS)

clean:
	rm -f m3 c/tests/test_msa

# C unit tests (kernels, MSA, etc.) — compile engine.c with -DTESTING to skip main()
c/tests/test_msa: c/tests/test_msa.c src/engine.c src/st.h src/json.h src/planar_kv.h
	@mkdir -p c/tests
	$(CC) -DTESTING $(CFLAGS) c/tests/test_msa.c -o c/tests/test_msa $(LDFLAGS)

test-c: c/tests/test_msa
	@echo "==> running C unit tests"
	./c/tests/test_msa

test-converter:
	@echo "==> running converter unit tests"
	python3 tools/test_convert.py

test: test-converter test-c

check:
	@echo "==> lint gate: clean build with -Wall -Wextra (C static analysis)"
	@make clean > /dev/null
	@build_log=$$(make 2>&1); \
	if echo "$$build_log" | grep -iE "warning:|error:|undefined reference|fatal"; then \
		echo "$$build_log"; \
		echo "FAIL: build emitted warnings/errors"; \
		exit 1; \
	else \
		echo "PASS: clean build, no warnings/errors"; \
	fi
