# Axiom Edge

Distributed proving infrastructure for [OpenVM](https://github.com/openvm-org/openvm) programs. A **manager** accepts proof requests and schedules work across GPU **workers**, which run app proving and recursive (leaf/internal) aggregation. The manager tracks proof state and returns the final aggregated proof.

It's designed for a **trusted private network** (a GPU cluster behind your own perimeter). Endpoints are unauthenticated by design — see [Architecture › Trust Boundary](docs/ARCHITECTURE.md) before exposing it.

## Setup

You'll need:

* **Rust** — pinned to **1.91.1** by [`rust-toolchain.toml`](./rust-toolchain.toml) (rustup installs it automatically on first build); minimum supported version **1.91.1**.
* **Docker** — Engine **≥ 23** (BuildKit) with **Compose v2** (the `docker compose` subcommand). GPU runs additionally need the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).
* **CUDA** *(real/GPU proving only — the [mock quickstart](#mock-quickstart-no-gpu) needs no GPU)* — built against **CUDA 13.1** inside the bundled `nvidia/cuda:13.1.*` images, so you don't install the CUDA toolkit on the host; you only need an NVIDIA **driver** new enough for CUDA 13.1 (**≥ 580**; tested on 590). The default build targets **Blackwell** (`cuda_arch = 120`, ≥ 16 GB VRAM) — change `cuda_arch` in [`config/defaults.toml`](./config/defaults.toml) for other GPUs.
* **uv** — recent version, to run [`scripts/dev/start-provers.py`](./scripts/dev/start-provers.py).

## Mock quickstart (no GPU)

The mock stack runs the full manager/worker control plane with mock provers —
no GPU, no proving artifacts, no input file. It ships with a placeholder
loadout (`mock-program@1`), so it works out of the box:

```sh
docker compose -f docker/docker-compose.mock.yml up --build
```

Once `curl -s localhost:3000/readyz` reports `{"ready":true,...}`, start a
proof and poll its state (mock provers return placeholder results in seconds):

```sh
curl -s -X POST localhost:3000/start_proof \
    -H 'Content-Type: application/json' \
    -d '{"proof_uuid":"demo-1","input_already_uploaded":true}'
curl -s localhost:3000/proof_state/demo-1
```

## Quickstart

Requires NVIDIA GPUs, an OpenVM guest ELF, and generated artifacts.
Axiom Edge is program-agnostic — you supply the program(s) via a loadout JSON
(`--programs`), an array of `{name, version, path}` objects pointing at your
compiled OpenVM ELFs:

```jsonc
// programs.json
[
  { "name": "my-program", "version": 0, "path": "~/elf/my-program.elf" }
]
```

`scripts/dev/start-provers.py` renders configs, regenerates keys, and brings up the stack:

```sh
./scripts/dev/start-provers.py 4 \
    --total-provers 4 \
    --regenerate \
    --programs programs.json
```

Once `curl -s localhost:3000/readyz` reports `{"ready":true,...}`, submit a STARK
proof with `scripts/ops/start-proof.sh` (it uploads the input, submits, and polls
to completion):

```sh
./scripts/ops/start-proof.sh --via-manager \
    --input ~/input/example.bin \
    --program my-program --version 0
```

`--input` takes a bincode `StdIn` `.bin`, a `.json` (converted on the fly), or a
single-element `.compact`. `--program`/`--version` are optional when the loadout
has exactly one program. Two input transports:

- **`--via-manager` (shown above)** — uploads the input to the manager, which
  fans it out to the workers. The general path; no assumptions about the host
  reaching worker ports.
- **default (omit the flag)** — uploads directly to each worker (compact, fast;
  the ethproofs path), then submits. Requires the worker ports to be reachable
  from the host (`--worker-port-base`).

### EVM proof

EVM proofs add a root + halo2 wrapping stage after the STARK pipeline. Two extras
over the stark quickstart.

First, generate the halo2 proving key once, host-side (>10GB output; needs the KZG
SRS files):

```sh
cargo build --release -p edge-worker --features evm-prove --bin halo2-keygen
./target/release/halo2-keygen --kzg-params-dir <SRS_DIR> --output-dir <HALO2_PK_DIR>
```

Then start the stack in a halo2 mode with that key mounted:

```sh
./scripts/dev/start-provers.py 4 \
    --halo2 full \
    --halo2-pk-path <HALO2_PK_DIR> \
    --regenerate \
    --programs programs.json
```

(`--halo2` selects the EVM-wrap mode: `none` (default, stark-only — no halo2),
`full` (every worker is eligible for the manager-dispatched root → halo2 EVM
step), or `dedicated` (isolate that step on the highest-id worker, its own GPU). `full`/`dedicated` append
`evm-prove` to the default features `cuda,jemalloc,parallel,aot,unprotected` —
no need to re-list them via `--features`.)

Submit with `proof_type=evm` — either `start-proof.sh --proof-type evm …`, or by
hand: upload the bincode `StdIn` to the manager, then start the proof (the
request carries no input path — see [API endpoints](docs/API_ENDPOINTS.md)):

```sh
# 1. stage the input on the manager (multipart `input` part)
curl -sf -X POST http://localhost:3000/upload_input/evm-1 -F input=@foo.bin
# 2. start the proof (references the uuid; input_already_uploaded defaults to false)
curl -sX POST http://localhost:3000/start_proof -H 'Content-Type: application/json' \
  -d '{"proof_uuid":"evm-1","proof_type":"evm","program":{"name":"my-program","version":0}}'
```

### Recursively Verifying STARK Proofs

Proving a guest that verifies another proof requires substantial VM execution,
and a more efficient way is to use [deferral framework](https://docs.openvm.dev/book/acceleration-using-extensions/deferral)
See **[Deferral](docs/DEFERRAL.md)** for the runnable self-recursion quickstart.

### Multiple machines

Run one **manager** host (which also runs its local GPU workers) and any number
of **worker-only** hosts that register against it. The manager listens on
`:3000`; open that port from the worker hosts and make sure they can reach the
manager's LAN IP.

Three invariants across every host:

- `--total-provers` = the **cluster-wide** GPU total, identical on every host.
- `--id-offset` = the number of GPUs on all *earlier* hosts, so each host owns a
  contiguous block of prover IDs (the manager host uses the default `0`).
- `--programs` must point at the **same loadout** everywhere, since the manager
  dispatches a proof only to workers that serve its program. Each host
  regenerates its own keys/vmexes from that loadout (deterministic, so they
  agree), so the ELF(s) referenced by `programs.json` must exist on every host.

Example — 2 machines × 4 GPUs (8 provers total), manager at `10.0.0.1`:

```sh
# Host A (manager + 4 workers, prover IDs 0–3)
./scripts/dev/start-provers.py 4 \
    --total-provers 8 \
    --programs programs.json \
    --regenerate

# Host B (worker-only, 4 GPUs, prover IDs 4–7)
./scripts/dev/start-provers.py 4 \
    --total-provers 8 \
    --id-offset 4 \
    --worker-only \
    --manager-url http://{MANAGER_IP}:3000 \
    --programs programs.json \
    --regenerate
```

Add more worker hosts the same way, bumping `--id-offset` by each host's GPU
count (host C with 4 GPUs → `--id-offset 8`, and so on).

## Docs

- [Concepts](docs/CONCEPTS.md) — proof pipeline, programs/loadout, artifacts, segments, recursion.
- [Architecture](docs/ARCHITECTURE.md) — components, data flow, trust boundary, failure model.
- [API Endpoints](docs/API_ENDPOINTS.md) — manager + worker HTTP surface and wire types.
- [Deferral](docs/DEFERRAL.md) — proof-of-proof (`verify_stark`): deferral keygen, caller-derived inputs, submitting a stark or EVM deferral job.

## Layout

```text
crates/
  protocol/                Public wire types + protocol version
  proof/                   Proof types + encode/decode (internal)
  telemetry/               Tracing / OpenTelemetry setup (internal)
  edge-manager/            HTTP manager: scheduler, proof state, aggregation
  edge-worker/             Worker HTTP service + prover pool
  edge-integration-tests/  Mock E2E + integration tests
config/                    Defaults, templates, testing configs
docker/                    Dockerfiles and compose files
scripts/                   Dev (start-provers.py) and ops helpers
docs/                      Concepts, Architecture, API endpoints
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
