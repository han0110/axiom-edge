# Axiom Edge API Endpoints

This document describes all HTTP API endpoints exposed by the Axiom Edge proving system.

## Overview

The Axiom Edge system consists of two service types:
- **Edge Manager** (port 3000): Orchestrates proof generation and receives results from workers
- **Edge Worker** (port 8001+): Worker nodes that execute proving tasks

---

## Edge Manager Endpoints (Port 3000)

### POST `/start_proof`

Start a new Edge proof request. This is the main entry point for clients.

**Request Body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `proof_uuid` | string | Yes | Unique identifier for this proof |
| `program` | object `{name: string, version: u32}` | Conditional | Target program from the deployment loadout. **Optional only when exactly one program is loaded** (the sole program is used). Required otherwise. |
| `proof_type` | string: `"stark"` (default) or `"evm"` | No | Requested final proof artifact. `"stark"` stops at the final internal (recursion) proof. `"evm"` appends the root → halo2 wrapping stage and yields an on-chain-verifiable proof — requires workers built with the `evm-prove` feature and a mounted halo2 proving key. |
| `labels` | object (string→string) | No | Opaque, deployment-defined key/value metadata. The edge never interprets it — forwarded in lifecycle webhook events and emitted as metric attributes for downstream integrations (a caller might set, e.g., `{"block_number": "24000000"}` or `{"batch_id": "…"}`). |
| `input_already_uploaded` | bool | No | Selects the input transport. `false` (default) — **manager-staged (Flow 2)**: the caller uploads the bincode `StdIn` to the manager (`POST /upload_input/{proof_uuid}`) first, and the manager fans it out to the workers. `true` — **worker pre-uploaded (Flow 1)**: the caller pushed the input directly to every worker (e.g. `/upload_input_compact`) first, so the manager skips fan-out. There is no caller-supplied path; the worker always reads `/dev/shm/edge_{proof_uuid}/input.bin`. |
| `segment_memory` | usize | No | Override for OPENVM_MAX_SEGMENT_MEMORY |
| `leaf_pack_threshold` | usize | No | Override leaf packing threshold (large value packs leaf proofs onto busy workers) |
| `timeout_secs` | u64 | No | Per-proof watchdog timeout override; falls through to the manager's `[proof] timeout_secs` if unset |

There is **no** deferral field on this request. Whether a proof is a deferral proof — and how many circuits — is inferred by the manager from the `DeferralState`/`DeferralInput` artifacts the caller staged on it beforehand (one pair per circuit, at contiguous indices `0..N`; see the upload endpoints below). Deferral requires the manager-staged transport (`input_already_uploaded=false`).

**Response:**

- **200 OK**: Proof started successfully
```json
{
  "proof_uuid": "...",
  "status": "started",
  "num_workers": 4
}
```

- **409 Conflict**: Another proof is already running, proof UUID already exists, or the requested program is not in the loadout
```json
{
  "error": "program_not_in_loadout",
  "message": "Program program1@v2 is not in the current loadout",
  "current_loadout": [{ "name": "program1", "version": 0 }]
}
```

- **400 Bad Request**: `program` omitted while the loadout holds anything other than exactly one program
```json
{
  "error": "program_required",
  "message": "Specify `program: {name, version}` in the request; the loadout does not hold exactly one program",
  "current_loadout": [{ "name": "program1", "version": 0 }, { "name": "program2", "version": 0 }]
}
```

- **503 Service Unavailable**: Workers not ready
```json
{
  "error": "Workers not ready: ..."
}
```

---

### POST `/upload_input/{proof_uuid}` (manager, Flow 2)

Stage **all** of a proof's input on the manager in one `multipart/form-data`
request, before `/start_proof`. The manager holds the parsed bytes in memory
keyed by `proof_uuid`. Parts (all optional):

| Part name | Contents |
|---|---|
| `input` | The bincode `StdIn` bytes (the main program input). |
| `input_compact` | The compact bytes for one logical input element, as an alternative to `input`. Each worker wraps them into a `StdIn`, so a caller with no OpenVM types on hand can still stage input on the manager. Rejected at `/start_proof` for a deferral proof, whose artifacts need a real `StdIn` to insert into. |
| `deferral_state_{i}` | Circuit `i`'s `DeferralState` (one per deferral circuit, contiguous `0..N`). Fanned out to every app worker at `/start_proof`. |
| `deferral_input_{i}` | Circuit `i`'s `DeferralInput`. The manager **retains** it (never broadcasts) and pushes it just-in-time to the worker that produces the final internal proof. |

A single call covers non-deferral (just `input`) and deferral (`input` + N
pairs) proofs alike — the caller never makes more than one upload request. Body
limit is disabled (parts can be large). Returns **200 OK**, or **400 Bad
Request** for an invalid `proof_uuid`, a malformed body, or an unexpected part
name.

Example (curl):

```sh
curl -sX POST http://localhost:3000/upload_input/my-proof \
  -F input=@stdin.bin \
  -F deferral_state_0=@state_0.bin \
  -F deferral_input_0=@input_0.bin
```

---

### POST `/proof_result`

Receive proof results from workers. Workers call this endpoint to report completed proofs.

**Request Body:** Bincode-serialized `ResultPayload`

```rust
struct ResultPayload {
    worker_id: usize,
    proof_uuid: String,
    result: MessageEnvelope<ProofResult>,
}
```

Where `ProofResult` can be:
- `ExecuteE2` - Execution metadata (segment count, cost)
- `App` - Single segment app proof
- `Leaf` - Aggregated leaf proof
- `Internal` - Internal recursion proof
- `Evm` - Final EVM proof (the `proof_type=evm` tail: root → halo2). There is no separate `Root` result; root timing folds into `Evm`.
- `Error` - Error report

**Response:**

- **200 OK**: Result accepted
- **400 Bad Request**: Invalid payload or proof_uuid mismatch
- **404 Not Found**: Unknown proof_uuid
- **500 Internal Server Error**: Processing failed

---

### POST `/register_worker`

Register a worker with the manager. Workers call this on startup and periodically re-register.

**Request Body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `worker_url` | string | Yes | HTTP URL where this worker can be reached |
| `worker_id` | usize | Yes | Stable worker ID (the worker's configured `prover_id`). The manager validates the URL↔ID binding. |
| `max_app_provers` | usize | Yes | App prover instances on this worker (also the per-proof app-prove parallelism). |
| `max_leaf_provers` | usize | Yes | Concurrent leaf proofs this worker can run. |
| `max_internal_provers` | usize | Yes | Concurrent internal proofs this worker can run. |
| `worker_role` | string: `"full"` (default), `"stark_only"`, or `"evm_dedicated"` | No | Deployment role. `full` runs every step; `stark_only` runs app/leaf/internal but no EVM step; `evm_dedicated` runs **only** the dispatched EVM step (must report zero app/leaf/internal capacity). Only the dedicated-halo2 deployment mode uses non-default roles. |
| `loaded_programs` | array of `{name, version}` | No | Programs this worker has loaded vmexes for. Advisory. The manager pushes every registered program absent from this list to the worker, so a late-joining or restarted worker converges on the current loadout without operator action. |

**Response:**

- **200 OK**: Worker registered
```json
{
  "status": "ok",
  "worker_id": 0
}
```

- **400 Bad Request**: Registration failed (e.g. a worker_id↔URL conflict)

---

### POST `/register_program`

Register a guest program with the deployment. The manager retains the payload
and pushes it to every registered worker, each of which answers with the
verifying key it derived. Those keys must agree, and the agreed one is cached
for `GET /program_vk/{name}/{version}`. Body limit is disabled (the ELF can be
large).

**Request Body (`multipart/form-data`):**

| Part name | Contents |
|---|---|
| `program` | JSON `{name, version}` this program is registered under. |
| `elf` | Raw guest ELF bytes. |
| `vm_config` | Serialized `SdkVmConfig`, opaque to the manager. |

Workers ahead-of-time compile the guest after answering, so a successful
registration does not mean the deployment can prove yet. `/start_proof` waits
for that compile and answers 503 if it takes longer than two minutes.

**Response:**

- **202 Accepted**: Pushed to every registered worker
```json
{
  "status": "registering",
  "program": { "name": "program1", "version": 0 },
  "num_workers": 4
}
```

- **200 OK**: Identical program, ELF, and config are already registered
```json
{
  "status": "unchanged",
  "program": { "name": "program1", "version": 0 }
}
```

- **400 Bad Request**: Malformed multipart body, an unexpected part, or a `missing_part`
- **409 Conflict**: `program_already_registered`, this `{name, version}` already holds different bytes
- **500 Internal Server Error**: `program_vk_mismatch`, workers derived different verifying keys
- **502 Bad Gateway**: `worker_rejected_program`, a worker was unreachable or refused the push
- **503 Service Unavailable**: `no_workers_registered`

A registration that fails is rolled back out of the loadout, so a retry starts
clean.

---

### GET `/program_vk/{name}/{version}`

Return the verification baseline the workers derived for a registered program.
This reads cached state, so it neither reaches out to a worker nor gates on
readiness. A client holding the key still cannot prove with it until every
worker has finished its AOT compile.

**Path Parameters:**
- `name`: The program name
- `version`: The program version

**Response:**

- **200 OK**: bincode `verify_stark::VerificationBaseline` (`application/octet-stream`)
- **404 Not Found**: `program_vk_unknown`, no worker has reported a key for this program
- **500 Internal Server Error**: `program_vk_mismatch`, workers disagree on the key

---

### GET `/loadout`

Return the manager's current program loadout, seeded from `EDGE_PROGRAMS` and extended by `/register_program`.

**Response (JSON):**

- **200 OK**:
```json
{
  "programs": [
    { "name": "program1", "version": 0 },
    { "name": "program2", "version": 0 }
  ]
}
```

---

### GET `/workers`

Query the current worker registration status.

**Response (JSON):**

Returns the current status of all registered workers, including their URLs and registration information.

---

### GET `/readyz`

Query whether the full Edge worker stack is ready for direct input uploads and proof start.

The manager uses declared `num_provers` registration metadata, checks that all expected
workers are registered, and probes each worker's `/readyz` endpoint before returning `200`.

**Response (JSON):**

- **200 OK**: Full worker stack is registered and ready
```json
{
  "ready": true,
  "num_workers": 16,
  "expected_num_workers": 16,
  "workers": [
    [0, { "worker_url": "http://10.0.0.1:8001", "last_seen": "2026-04-08T00:00:00Z", "worker_role": "full" }]
  ]
}
```

- **503 Service Unavailable**: Stack is incomplete or at least one worker is not ready
```json
{
  "ready": false,
  "num_workers": 15,
  "expected_num_workers": 16,
  "message": "Only 15/16 Edge workers have registered with manager",
  "workers": []
}
```

---

### GET `/proof_state/{proof_uuid}`

Get the current state of a proof.

**Path Parameters:**
- `proof_uuid`: The proof identifier

**Response:**

- **200 OK**: Lightweight proof state
```json
{
  "proof_uuid": "...",
  "program": { "name": "program1", "version": 0 },
  "status": "in_progress",
  "num_segments": 16,
  "num_instructions": 16000000,
  "proof_start_time": "2024-01-01T00:00:00Z",
  "e2e_latency_ms": null,
  "app_proofs_count": 8,
  "leaf_proofs_count": 2,
  "internal_proofs_count": 0,
  "last_updated": "2024-01-01T00:00:30Z"
}
```

- **404 Not Found**: Proof not found

---

### GET `/final_proof/{proof_uuid}`

Return the bytes of a completed STARK proof, decompressed whether or not the
deployment set `proof.compress_persisted_final_proofs`. The path is derived
from `proof.persist_final_proofs_dir` rather than from proof state, so a proof
stays fetchable after its terminal state is evicted.

**Path Parameters:**
- `proof_uuid`: The proof identifier

**Response:**

- **200 OK**: openvm-codec-encoded `verify_stark::VmStarkProof` (`application/octet-stream`)
- **400 Bad Request**: Invalid `proof_uuid`
- **404 Not Found**: `final_proof_not_found`
- **409 Conflict**: `final_proofs_not_persisted`, the deployment configured no `proof.persist_final_proofs_dir`
- **500 Internal Server Error**: `final_proof_unreadable`

---

### GET `/proof_debug/{proof_uuid}`

Get scheduler-side per-worker debug state for an in-progress proof.

This endpoint is intended for stall diagnosis and exposes worker progress as tracked
by manager scheduling logic.

**Path Parameters:**
- `proof_uuid`: The proof identifier

**Response:**

- **200 OK**: Scheduler debug state
```json
{
  "proof_uuid": "...",
  "num_workers": 16,
  "num_segments": 124,
  "pending_work_empty": false,
  "workers": [
    {
      "worker_id": 6,
      "worker_url": "http://10.0.0.7:8003",
      "active_proof_count": 1,
      "active_steps": ["sharded_app_prove"],
      "completed_segments_received": 5,
      "completed_segments_mod_match": 5,
      "expected_segments_mod_match": 8,
      "remaining_segments_mod_match": 3
    }
  ]
}
```

- **404 Not Found**: Proof not found

---

### POST `/cancel_proof`

Cancel an in-progress proof.

**Request Body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `proof_uuid` | string | Yes | The proof identifier to cancel |

**Response:**

- **200 OK**: Cancellation initiated
```json
{
  "status": "canceled"
}
```

---

### GET `/healthz`

Health check endpoint.

**Response:**

- **200 OK**:
```json
{
  "status": "healthy"
}
```

---

## Edge Worker Endpoints (Port 8001+)

### POST `/register_program`

Load a guest program into this worker. Called by the manager when it fans out a
`/register_program` request, and when it replays the loadout to a worker that
has just registered.

**Request Body:** Bincode-serialized `RegisterProgramRequest`

```rust
struct RegisterProgramRequest {
    program: ProgramRef,
    elf: Vec<u8>,
    vm_config: String,  // serialized SdkVmConfig
}
```

The worker transpiles the ELF into a `VmExe`, runs app and aggregation keygen
against `vm_config`, and answers once the verifying key is derived. The AOT
compile and GPU prover preload continue after the response, so `/readyz` is
what reports when the program is servable. The first registration pins the
worker's VM config, so serving a different config takes a restart rather than
another registration.

**Response:**

- **200 OK**: bincode `verify_stark::VerificationBaseline` (`application/octet-stream`), empty on a mock build, which derives no keys
- **400 Bad Request**: Undeserializable body or a `vm_config` that is not a valid `SdkVmConfig`
- **409 Conflict**: Incompatible with what this worker already serves

---

### POST `/upload_input/{proof_uuid}`

Upload input data for a proof. Called by the manager when it fans out a Flow-2 (manager-staged) input, or by an external producer pre-uploading directly to the workers for Flow 1 (`input_already_uploaded=true`).

**Path params:** `proof_uuid`.

**Request Body:** the raw binary input bytes (bincode `StdIn`) — no framing.

**Behavior:**
1. Creates work directory at `/dev/shm/edge_{proof_uuid}`
2. Writes input to `/dev/shm/edge_{proof_uuid}/input.bin` (atomic two-phase write)

**Validation:**
- proof_uuid must be non-empty ASCII alphanumeric plus `_` and `-` (allowlist; anything else — including `/`, `\`, `.` — is rejected, preventing path traversal)
- proof_uuid maximum length: 256 characters

**Response:**

- **200 OK**: `"Input file received"`
- **400 Bad Request**: Invalid format or proof_uuid
- **500 Internal Server Error**: Failed to create/write file

---

### POST `/upload_input_compact/{proof_uuid}`

Upload compact guest-input bytes for a proof. This is the Flow-1 direct-upload endpoint for a high-throughput producer co-located with the workers (single logical input element; the worker wraps it into a `StdIn`).

**Path params:** `proof_uuid`.

**Request Body:** the raw compact guest-input bytes — no framing (the worker wraps them into a `StdIn`).

**Behavior:**
1. Creates work directory at `/dev/shm/edge_{proof_uuid}`
2. Converts the compact guest-input bytes into `bincode(StdIn<F>)` locally on the worker
3. Writes the resulting `input.bin` to `/dev/shm/edge_{proof_uuid}/input.bin` (atomic two-phase write)

**Validation:**
- proof_uuid must be non-empty ASCII alphanumeric plus `_` and `-` (allowlist; anything else — including `/`, `\`, `.` — is rejected, preventing path traversal)
- proof_uuid maximum length: 256 characters
- the compact guest-input payload must be valid for worker-side conversion into `StdIn<F>`

**Response:**

- **200 OK**: `"Input file received"`
- **400 Bad Request**: Invalid format, proof_uuid, or compact input payload
- **500 Internal Server Error**: Failed to create/write file

---

### POST `/upload_deferral_state/{proof_uuid}`

Stage **all** circuits' caller-derived `DeferralState`s on this worker in one
call, before app proving starts. Called by the manager's Flow-2 fan-out (the
caller uploaded the states to the manager as `deferral_state_{i}` multipart
parts); the payload is opaque to the edge.

**Path params:** `proof_uuid` (same allowlist validation as `/upload_input`).

**Request Body:** bincode `Vec<Vec<u8>>` — index = circuit. The worker
validates the count against its loaded deferral keyset and writes each entry
to `/dev/shm/edge_{proof_uuid}/deferral_state_{i}.bin`.

**Response:**

- **200 OK**: `"Deferral artifacts received"`
- **400 Bad Request**: invalid proof_uuid, malformed payload, or a count that doesn't match the worker's deferral keyset
- **500 Internal Server Error**: failed to write an artifact file

---

### POST `/upload_deferral_input/{proof_uuid}`

Stage **all** circuits' caller-derived `DeferralInput`s on this worker in one
call. Unlike `DeferralState`s (broadcast to every app worker at proof start),
the manager retains `DeferralInput`s and pushes them **just-in-time** to the
single worker that will produce the final internal proof, right before that
dispatch.

**Path params:** `proof_uuid` (same allowlist validation as `/upload_input`).

**Request Body:** bincode `Vec<Vec<u8>>` — index = circuit. The worker
validates the count against its loaded deferral keyset and writes each entry
to `/dev/shm/edge_{proof_uuid}/deferral_input_{i}.bin`.

**Response:**

- **200 OK**: `"Deferral artifacts received"`
- **400 Bad Request**: invalid proof_uuid, malformed payload, or a count that doesn't match the worker's deferral keyset
- **500 Internal Server Error**: failed to write an artifact file

---

### POST `/sharded_app_prove`

Execute sharded app proving (execution + app proving) for this worker's assigned segments.

**Request Body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `proof_uuid` | string | Yes | Proof identifier |
| `program` | object `{name: string, version: u32}` | Yes | Target program; used by the worker to select the matching vmexe and (lazily) load the GPU prover for that program |
| `prover_id` | usize | Yes | This worker's ID (0-indexed) |
| `num_provers` | usize | Yes | Total number of workers |
| `segment_memory` | usize | No | Override segment memory |

The request carries **no input path**: the worker reads its input (and any
staged `DeferralState`s) from the deterministic staged location
`/dev/shm/edge_{proof_uuid}/input.bin`, populated by the manager fan-out in
Flow 2 or by the caller's direct upload in Flow 1.

**Behavior:**
1. Waits for input file to exist (30-second timeout)
2. Checks for available worker in prover pool
3. Spawns proving task in background
4. Returns immediately (non-blocking)
5. Results sent back via POST to manager's `/proof_result`

**Response:**

- **200 OK**: `"Edge work accepted"`
- **408 Request Timeout**: `"Timeout waiting for input file: {path}"`
- **503 Service Unavailable**: `"No available app workers"`

---

### POST `/recursion_prove`

Execute leaf or internal proof work.

**Request Body:** Bincode-serialized `MessageEnvelope<GeneralProveRequest>`

Where `GeneralProveRequest` can be:
- `LeafProve(LeafProveRequest)` - Aggregate app proofs into leaf proof
- `InternalProve(InternalProveRequest)` - Aggregate leaf/internal proofs
- `EvmProve(EvmProveRequest)` - Run the EVM step (root → halo2) on a finished internal proof. Dispatched by the manager in **every** deployment mode once the final internal proof (plus any deferral tail merge) is done: to any `Full` worker in the default deployment, or to the `EvmDedicated` worker in dedicated-halo2 mode. The EVM step never runs in-process on the final-internal worker without a dispatch.

**LeafProveRequest fields:**
- `context`: ProofContext
- `app_proofs`: Vec of app proofs to aggregate
- `segment_start`: First segment index
- `segment_end`: Last segment index (inclusive)

**InternalProveRequest fields:**
- `context`: ProofContext
- `child_proofs`: Vec of child proofs to aggregate
- `layer_idx`: Recursion tree layer index (0 = first internal layer)
- `segment_start`: First segment index
- `segment_end`: Last segment index (inclusive)
- `is_final_proof`: Whether this is the final proof in the tree
- `deferral_tail`: *(optional)* manager → tail-worker handoff attached to the **final** internal prove of a deferral job, sequencing the deferral merge (`prove_def → prove_mixed → wrap`) before root. Absent for non-deferral proofs.
- `deferral_merkle_proofs_bytes`: *(optional)* encoded depth-0 `DeferralMerkleProofs` for a proof that made no deferred calls on a deferral deployment. Mutually exclusive with `deferral_tail`; absent otherwise.

**EvmProveRequest fields:**
- `context`: ProofContext
- `internal_proof_bytes`: the finished internal proof (bincode-encoded `ProofWithPublicValue`)
- `deferral_merkle_proofs_bytes`: *(optional)* serialized deferral merkle proofs attached to root's `VmStarkProof`; absent on a non-deferral deployment
- `proof_has_deferral`: whether this proof ran the deferral tail merge

**Response:**

- **200 OK**: `"Work completed"`
- **400 Bad Request**: Invalid payload or unexpected request type
- **503 Service Unavailable**: No available workers

---

### GET `/healthz`

Health check endpoint with worker status.

**Response:**

- **200 OK**:
```json
{
  "status": "ok",
  "app_workers_busy": 2,
  "leaf_workers_busy": 0,
  "internal_workers_busy": 1
}
```

---

### GET `/readyz`

Readiness check endpoint (artifacts loaded). `programs` lists what this worker
serves, which the manager checks the target program against before dispatching,
since a worker can be ready in general while missing a program whose push never
reached it.

**Response:**

- **200 OK**: Worker is ready
```json
{
  "ready": true,
  "message": "Worker is ready",
  "programs": [{ "name": "program1", "version": 0 }]
}
```

- **503 Service Unavailable**: Not ready
```json
{
  "ready": false,
  "message": "Artifacts not loaded",
  "programs": []
}
```

---

## Data Types

### MessageEnvelope

Wraps all messages with metadata for idempotent delivery:

```rust
struct MessageEnvelope<T> {
    timestamp: u64,      // Unix timestamp in ms
    message_id: String,  // UUID for idempotency
    message: T,          // Actual payload
}
```

### ProofContext

Shared metadata for all proof operations:

```rust
struct ProofContext {
    proof_uuid: String,
    program: ProgramRef,                  // { name: String, version: u32 }
    labels: BTreeMap<String, String>,     // opaque deployment metadata
    proof_type: ProofType,                // "stark" (default) | "evm"
}
```

### ProgramRef

Identifies one program version in the deployment loadout:

```rust
struct ProgramRef {
    name: String,
    version: u32,
}
```

Rendered as `name@vversion` in logs (e.g. `program1@v0`).

### ProofStatus

Possible proof states:

| Status | Description |
|--------|-------------|
| `in_progress` | Proof is currently being generated |
| `completed` | Proof completed successfully |
| `failing` | A worker reported a fatal error; the manager is draining peer workers before settling into `failed` (transient, non-terminal) |
| `failed` | Proof failed with error message |
| `canceled` | Proof was canceled |

---

## How APIs being used in the Flow

Flow 2 — manager-staged (default; the only path for deferral):

```
1. Client POST /upload_input/{uuid} -> Manager (one multipart request: input + any deferral_state_{i}/deferral_input_{i} parts)
2. Client POST /start_proof (input_already_uploaded=false) -> Manager
3. Manager fans the input out to all workers via POST /upload_input (and, for a
   deferral proof, all DeferralStates via POST /upload_deferral_state)
4. Manager sends work requests to workers via POST /sharded_app_prove
5. Workers execute proving tasks and POST results to Manager via /proof_result
6. Manager triggers leaf/internal proofs via POST /recursion_prove; for a deferral
   proof it pushes each DeferralInput to the final-internal worker just before that dispatch
7. Repeat until the final proof is generated
8. Client queries /proof_state/{uuid} to get completion status
```

Flow 1 — worker pre-uploaded (fast path for a producer co-located with the workers):

```
1. Client GET /readyz -> Manager
2. Client uploads the compact guest input to all ready workers via POST /upload_input_compact
3. Client POST /start_proof with input_already_uploaded=true -> Manager
4. Manager sends work requests to workers via POST /sharded_app_prove
5. Workers execute proving tasks and POST results to Manager via /proof_result
6. Client polls /proof_state/{uuid} until a terminal status is reached
```

---

## Error Handling

All endpoints follow consistent error response patterns:

- **400 Bad Request**: Invalid input, malformed payload
- **404 Not Found**: Unknown proof_uuid or resource
- **408 Request Timeout**: Timeout waiting for resources
- **409 Conflict**: Resource conflict (duplicate proof, concurrent proof running)
- **500 Internal Server Error**: Processing failures
- **503 Service Unavailable**: Service not ready (workers unavailable, artifacts not loaded)

Error responses include an `error` field with a descriptive message:

```json
{
  "error": "Description of what went wrong"
}
```
