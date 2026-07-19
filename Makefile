CC ?= gcc
ARCH ?= native
CFLAGS = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-function -Isrc
LDFLAGS = -lm -fopenmp

# All C unit-test binaries. Each compiles engine.c with -DTESTING (which skips
# the engine's main()) and links against the test's own main().
C_TESTS = c/tests/test_msa \
          c/tests/test_matmul_i4 \
          c/tests/test_rope \
          c/tests/test_rmsnorm \
          c/tests/test_swiglu \
          c/tests/test_moe_routing \
          c/tests/test_per_head_qk_norm \
          c/tests/test_kv_quant \
          c/tests/test_expert_slab_load

# Engine + headers shared by every C test.
ENGINE_SRCS = src/engine.c src/st.h src/json.h src/planar_kv.h

.PHONY: all clean m3 test test-c test-tokenizer test-converter check

all: m3

m3: src/engine.c src/st.h src/json.h src/planar_kv.h
	$(CC) $(CFLAGS) src/engine.c -o m3 $(LDFLAGS)

clean:
	rm -f m3 $(C_TESTS)

# Pattern rule: any c/tests/test_*.c compiles to c/tests/test_* with -DTESTING.
c/tests/test_%: c/tests/test_%.c $(ENGINE_SRCS)
	@mkdir -p c/tests
	$(CC) -DTESTING $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(C_TESTS)
	@echo "==> running C unit tests"
	@for t in $(C_TESTS); do \
		echo "--- $$t ---"; \
		./$$t || exit 1; \
	done

test-tokenizer:
	@echo "==> running tokenizer round-trip tests"
	@# Prefer a venv that has the tokenizers lib; fall back to system python3.
	@if [ -x /home/ai/llama-convert-venv/bin/python ]; then \
		PY=/home/ai/llama-convert-venv/bin/python; \
	else \
		PY=python3; \
	fi; \
	$$PY c/tests/test_tokenizer.py

test-converter:
	@echo "==> running converter unit tests"
	@python3 tools/test_convert.py

test: test-converter test-c test-tokenizer

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
