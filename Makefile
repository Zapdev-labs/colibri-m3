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
          c/tests/test_expert_slab_load \
          c/tests/test_vnni \
          c/tests/test_numa \
          c/tests/test_int8_kv

# Engine + headers shared by every C test.
ENGINE_SRCS = src/engine.c src/st.h src/json.h src/planar_kv.h src/vnni.h src/numa.h src/observability.h

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
	@if [ -x python3 ]; then \
		PY=python3; \
	else \
		PY=python3; \
	fi; \
	$$PY c/tests/test_tokenizer.py

test-converter:
	@echo "==> running converter unit tests"
	@python3 tools/test_convert.py

# f8-oracle-validation: oracle comparison + end-to-end correctness tests.
# These drive the full engine on the 199GB int4 snapshot; each takes 5-15 min
# (model load dominates). Run individually or via `make test-oracle-all`.
test-oracle-logits:
	@echo "==> running oracle teacher-forcing logits comparison (VAL-CORR-014)"
	@python3 tools/test_oracle_logits.py

test-oracle-greedy:
	@echo "==> running oracle greedy decode comparison (VAL-CORR-015)"
	@python3 tools/test_oracle_greedy.py

test-eos:
	@echo "==> running EOS honoring test (VAL-CORR-017)"
	@python3 tools/test_eos.py

test-determinism:
	@echo "==> running seeded determinism test (VAL-CORR-018, VAL-CROSS-006)"
	@python3 tools/test_determinism.py

test-kv-cache:
	@echo "==> running KV cache correctness test (VAL-CORR-019)"
	@python3 tools/test_kv_cache.py

test-nan:
	@echo "==> running numerical stability test (VAL-CORR-024)"
	@python3 tools/test_numerical_stability.py

# f9: cross-oracle comparison vs llama.cpp fork (VAL-CORR-021).
# Requires the 247GB GGUF model and llama-server; gated behind explicit invocation.
test-cross-oracle:
	@echo "==> running cross-oracle comparison vs llama.cpp fork (VAL-CORR-021)"
	@python3 tools/cross_oracle_compare.py --mode both --ngen 20 --tf-tokens 32

# f13: throughput benchmark targeting >=5 tok/s. Requires the 199GB int4 model.
test-throughput:
	@echo "==> running f13 throughput benchmark (target >=5 tok/s)"
	@python3 tools/bench_throughput.py --ngen 200 --use-vnni --numa-interleave

# f15: cross-engine observability harness (requires both engines + telemetry).
test-observability:
	@echo "==> running f15 cross-engine observability harness"
	@python3 tools/observability.py --compare

# f9/f13 lightweight smoke tests (no 247GB model load required).
test-cross-oracle-smoke:
	@echo "==> running f9 cross-oracle harness smoke test"
	@python3 tools/test_cross_oracle.py

test-throughput-smoke:
	@echo "==> running f13 throughput benchmark smoke test"
	@python3 tools/test_throughput.py

test-oracle-all: test-oracle-greedy test-oracle-logits test-eos test-determinism test-kv-cache test-nan

test: test-converter test-c test-tokenizer test-cross-oracle-smoke test-throughput-smoke
	@echo "==> (oracle / e2e tests are gated behind 'make test-oracle-all' because they"
	@echo "     each reload the 199GB model; run them explicitly when the host is free)"

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
