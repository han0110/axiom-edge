#!/usr/bin/env bash

set -euo pipefail

# Reproduces a GPU out-of-memory failure in the VPMM allocator.
#
# The pool commits physical pages and never releases them. `free` returns a
# region to the pool's own free list, and `vpmm_release` is reachable only from
# the allocation-failure rollback and from `Drop`, which never runs because the
# memory manager is a `static OnceLock` built under `#[ctor]`. The pool's size
# is therefore the high-water mark of concurrent live bytes and only ever grows.
#
# Allocations below the 16 MiB page size take a separate `cudaMallocAsync` path
# that must obtain memory from the driver. Once the pool has ratcheted, that
# path starves and a multi-megabyte request fails with
# `cudaErrorMemoryAllocation` while tens of gigabytes sit idle in the pool.
#
# A single proof does not fail, since the pool grows alongside the small-
# allocation path while both still fit. The failure needs at least two proofs in
# one worker process, so each input is proved ITERATIONS times in sequence.
#
# The inputs in fixtures/inputs are compact guest bytes, which the worker wraps
# into a `StdIn` itself. They are derived from the EEST blockchain test of the
# same name through the zkevm-benchmark stateless-witness pipeline, which lives
# outside this repository, so they ship pre-converted.

# The repository root is the working directory for the whole run, and every
# path below is relative to it, so the generated configs and logs carry nothing
# specific to the machine that produced them.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FIXTURES_DIR="fixtures"
WORK_DIR="${FIXTURES_DIR}/work"
ARTIFACTS_DIR="${WORK_DIR}/artifacts"
RUN_DIR="${WORK_DIR}/run"
MANAGER_LOG="${FIXTURES_DIR}/manager.log"
WORKER_LOG="${FIXTURES_DIR}/worker.log"

PROGRAM_NAME="stateless-validator-reth"
PROGRAM_VERSION=1
EDGE_PROGRAMS="[{\"name\":\"${PROGRAM_NAME}\",\"version\":${PROGRAM_VERSION}}]"
MANAGER_PORT=3000
WORKER_PORT=8001
MANAGER_URL="http://127.0.0.1:${MANAGER_PORT}"

FEATURES="cuda,parallel,aot,jemalloc"
READINESS_TIMEOUT=1800
PROOF_TIMEOUT=3600

# Number of times each fixture is proved against the same worker.
ITERATIONS="${ITERATIONS:-4}"
CUDA_ARCH="${CUDA_ARCH:-}"
RUST_LOG="${RUST_LOG:-info}"
MANAGER_PID=""
WORKER_PID=""

log() {
    echo "[repro] $*"
}

cleanup() {
    for pid in "${WORKER_PID}" "${MANAGER_PID}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2> /dev/null; then
            kill "${pid}" 2> /dev/null || true
            wait "${pid}" 2> /dev/null || true
        fi
    done
}
trap cleanup EXIT

for cmd in cargo curl nvidia-smi; do
    if ! command -v "${cmd}" > /dev/null; then
        echo "error: ${cmd} is required but not installed" >&2
        exit 1
    fi
done

shopt -s nullglob
INPUTS=("${FIXTURES_DIR}"/inputs/*.bin)
shopt -u nullglob
if [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "error: no compact guest input found in ${FIXTURES_DIR}/inputs" >&2
    exit 1
fi

if [[ -z "${CUDA_ARCH}" ]]; then
    # The build script shells out to nvidia-smi when CUDA_ARCH is unset, which
    # fails in containers without the query. Resolve it here instead.
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
    if [[ -z "${CUDA_ARCH}" ]]; then
        echo "error: could not detect the GPU compute capability; set CUDA_ARCH" >&2
        exit 1
    fi
fi
log "building for CUDA arch ${CUDA_ARCH} (this takes a while on a cold cache)"
export CUDA_ARCH
cargo build --release --features "${FEATURES}" \
    -p edge-manager -p edge-worker --bin edge-manager --bin edge-worker --bin convert_fixtures

BIN_DIR="target/release"

# Keygen is deterministic and expensive, so it is skipped once its outputs
# exist. Delete fixtures/work to force a regeneration.
VMEXE_PATH="${ARTIFACTS_DIR}/programs/${PROGRAM_NAME}/${PROGRAM_VERSION}/program.vmexe"
if [[ ! -f "${ARTIFACTS_DIR}/app_pk" || ! -f "${ARTIFACTS_DIR}/agg_stark_pk" || ! -f "${VMEXE_PATH}" ]]; then
    log "generating proving keys and the guest vmexe"
    rm -rf "${ARTIFACTS_DIR}"
    mkdir -p "${ARTIFACTS_DIR}"
    "${BIN_DIR}/convert_fixtures" keygen \
        --elf "${FIXTURES_DIR}/stateless-validator-reth-openvm.elf" \
        --output-dir "${ARTIFACTS_DIR}"
    mkdir -p "$(dirname "${VMEXE_PATH}")"
    mv "${ARTIFACTS_DIR}/program.vmexe" "${VMEXE_PATH}"
else
    log "reusing the artifacts already in ${ARTIFACTS_DIR}"
fi

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}/proofs"

cat > "${RUN_DIR}/manager.toml" << EOF
[server]
listen_addr = "0.0.0.0:${MANAGER_PORT}"
num_workers = 1

[provers]
max_app_provers = 2
max_leaf_provers = 2
max_internal_provers = 1

[proof]
leaf_arity = 4
internal_arity = 3
leaf_pack_threshold = 1000
timeout_secs = ${PROOF_TIMEOUT}
persist_final_proofs_dir = "${RUN_DIR}/proofs"

[telemetry]
log_level = "info"
EOF

cat > "${RUN_DIR}/worker.toml" << EOF
[server]
listen_addr = "0.0.0.0:${WORKER_PORT}"

[worker]
prover_id = 0
num_provers = 1
worker_url = "http://127.0.0.1:${WORKER_PORT}"
manager_url = "${MANAGER_URL}"

[artifacts]
artifacts_path = "${ARTIFACTS_DIR}"

[provers]
max_app_provers = 2
max_leaf_provers = 2
max_internal_provers = 1

[telemetry]
log_level = "info"
EOF

log "starting the manager, logging to ${MANAGER_LOG}"
EDGE_PROGRAMS="${EDGE_PROGRAMS}" RUST_LOG="${RUST_LOG}" NO_COLOR=1 \
    "${BIN_DIR}/edge-manager" --config "${RUN_DIR}/manager.toml" > "${MANAGER_LOG}" 2>&1 &
MANAGER_PID=$!

until curl -fsS "${MANAGER_URL}/healthz" > /dev/null 2>&1; do
    if ! kill -0 "${MANAGER_PID}" 2> /dev/null; then
        echo "error: the manager exited during startup, see ${MANAGER_LOG}" >&2
        exit 1
    fi
    sleep 1
done

log "starting the worker, logging to ${WORKER_LOG}"
# VPMM_PAGE_SIZE is the `[cuda]` default from config/defaults.toml. It sets the
# boundary between the two allocators, so it decides which requests take the
# starving `cudaMallocAsync` path. Left unset, the pool falls back to the device
# minimum granularity and the failing requests would be served by the pool.
EDGE_PROGRAMS="${EDGE_PROGRAMS}" VPMM_PAGE_SIZE=16777216 RUST_LOG="${RUST_LOG}" NO_COLOR=1 \
    "${BIN_DIR}/edge-worker" --config "${RUN_DIR}/worker.toml" > "${WORKER_LOG}" 2>&1 &
WORKER_PID=$!

log "waiting for the worker to build its provers"
elapsed=0
until curl -fsS "http://127.0.0.1:${WORKER_PORT}/readyz" > /dev/null 2>&1; do
    if ! kill -0 "${WORKER_PID}" 2> /dev/null; then
        echo "error: the worker exited during startup, see ${WORKER_LOG}" >&2
        exit 1
    fi
    if [[ ${elapsed} -ge ${READINESS_TIMEOUT} ]]; then
        echo "error: the worker was not ready within ${READINESS_TIMEOUT}s, see ${WORKER_LOG}" >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
log "worker ready after ${elapsed}s"

# Reports the GPU memory the pool holds versus the bytes actually live, the two
# numbers whose divergence is the failure.
pool_state() {
    grep -a "GPU mem:" "${WORKER_LOG}" | tail -1 \
        | grep -oE "current=[0-9.]+ [GM]iB.*in pool=[0-9.]+ [GM]iB" || true
}

failed=0
index=0
for input in "${INPUTS[@]}"; do
    name="$(basename "${input}" .bin)"
    for iteration in $(seq 1 "${ITERATIONS}"); do
        index=$((index + 1))
        uuid="repro-$(printf '%02d' ${index})"

        log "proof ${index}: ${name} (iteration ${iteration}/${ITERATIONS})"
        curl -fsS -X POST "http://127.0.0.1:${WORKER_PORT}/upload_input_compact/${uuid}" \
            -H 'Content-Type: application/octet-stream' \
            --data-binary "@${input}" > /dev/null

        curl -fsS -X POST "${MANAGER_URL}/start_proof" \
            -H 'Content-Type: application/json' \
            -d "{\"proof_uuid\":\"${uuid}\",\"program\":{\"name\":\"${PROGRAM_NAME}\",\"version\":${PROGRAM_VERSION}},\"input_already_uploaded\":true}" \
            > /dev/null

        # Poll until the manager reports a terminal state. `ProofStatus` is
        # `rename_all = "snake_case"`, so unit variants serialize as plain
        # strings and the two carrying a reason as single-key objects. `failing`
        # is not terminal, it precedes `failed`.
        while true; do
            state="$(curl -fsS "${MANAGER_URL}/proof_state/${uuid}" || true)"
            case "${state}" in
                *'"status":"completed"'*)
                    log "proof ${index} completed. $(pool_state)"
                    break
                    ;;
                *'"status":{"failed"'* | *'"status":"canceled"'*)
                    log "proof ${index} FAILED. $(pool_state)"
                    failed=1
                    break
                    ;;
            esac
            if ! kill -0 "${WORKER_PID}" 2> /dev/null; then
                echo "error: the worker died mid-proof, see ${WORKER_LOG}" >&2
                exit 1
            fi
            sleep 5
        done

        if [[ ${failed} -eq 1 ]]; then
            break 2
        fi
    done
done

echo
if [[ ${failed} -eq 1 ]] && grep -aq "cudaErrorMemoryAllocation" "${WORKER_LOG}"; then
    log "reproduced the out-of-memory failure on proof ${index}"
    echo
    echo "Refused allocations, taking the cudaMallocAsync path below the 16 MiB page size:"
    grep -aoE "cudaMallocAsync failed: size=[0-9]+" "${WORKER_LOG}" | sort -u | sed 's/^/  /'
    echo
    echo "Pool state when it failed, committed versus live:"
    pool_state | sed 's/^/  /'
    echo
    echo "Worker error lines:"
    grep -a "cudaErrorMemoryAllocation" "${WORKER_LOG}" | tail -3 | cut -c1-200 | sed 's/^/  /'
    echo
    echo "Full logs are in ${MANAGER_LOG} and ${WORKER_LOG}."
    exit 1
elif [[ ${failed} -eq 1 ]]; then
    log "proof ${index} failed for a reason other than the allocator, see ${WORKER_LOG}"
    exit 1
else
    log "every proof completed, the pool did not ratchet far enough to fail"
    log "raise ITERATIONS or check ${WORKER_LOG} for the pool high-water"
    exit 0
fi
