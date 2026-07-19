CC ?= gcc
ARCH ?= native
CFLAGS = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-function -Isrc
LDFLAGS = -lm -fopenmp

.PHONY: all clean m3 test check

all: m3

m3: src/engine.c src/st.h src/json.h src/planar_kv.h
	$(CC) $(CFLAGS) src/engine.c -o m3 $(LDFLAGS)

clean:
	rm -f m3

test:
	@echo "==> running converter unit tests"
	python3 tools/test_convert.py

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
