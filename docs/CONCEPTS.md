# Axiom Edge Concepts

## Proof

A proof request identifies:

- `proof_uuid`: caller-provided proof ID
- `program`: `{name, version}` selecting one entry from the deployment loadout. Optional only when exactly one program is loaded.
- `labels`: optional opaque key/value metadata, forwarded to lifecycle webhooks and metrics (a caller might set, e.g., `block_number` or `batch_id` — the edge never interprets them)
- input bytes: uploaded before `/start_proof` (to the manager, or pre-uploaded to the workers) — see [Input Upload](#input-upload); not a field on the request

## Programs and loadout

A deployment's loadout is populated at runtime. A client posts a guest ELF, plus the VM config to build it under, to `/register_program`, and the manager fans that registration out to every worker and replays it to workers that register later. The `EDGE_PROGRAMS` env var, a JSON array of `{name, version}` objects parsed at startup by both the manager and every worker, optionally seeds the loadout with programs whose artifacts are already staged on the workers' disks. The manager rejects `/start_proof` requests for programs outside the loadout.

## Artifacts

Workers load proving artifacts before accepting work. The keys are **shared across all programs**; only the executable is per-program:

```text
{artifacts_path}/app_pk                                  # shared proving key
{artifacts_path}/agg_stark_pk                            # shared aggregation key
{artifacts_path}/programs/{name}/{version}/program.vmexe # per-program executable
```

All artifacts must be generated from the same OpenVM version and feature set, and each program's vmexe must come from the same OpenVM build as the keys. Artifact skew is a common source of worker readiness and errors.

## Worker

A worker is an edge worker process. It registers a deterministic `prover_id` with the manager and owns long-lived prover threads:

- app provers execute the program and prove assigned segments
- leaf provers aggregate app proofs
- internal provers aggregate leaf or internal proofs

## Segment

Execution splits a program run into segments. Segment ownership is deterministic:

```text
segment_idx % num_workers == prover_id
```

This makes app proving parallel without a manager-side segment queue.

## Recursion

The manager builds a recursion tree from worker results:

1. App proofs prove individual execution segments.
2. Leaf proofs aggregate batches of app proofs.
3. Internal proofs aggregate leaf or internal proofs.
4. The final internal proof is the completed edge proof.

The leaf batch size is configurable. Internal aggregation uses a fixed arity in the current implementation.

## Input Upload

Input bytes are always **uploaded** before `/start_proof` — the request body
carries no filesystem paths, only a transport flag (`input_already_uploaded`).
Whether a proof is a deferral proof, and how many circuits, is inferred from the
artifacts the caller uploaded (not declared on the request). A proof can involve
three kinds of input artifact:

| Artifact | Format | Consumer |
|---|---|---|
| Main program input | `bincode(StdIn<F>)` | Every app worker |
| `DeferralState` | Bincode, one per deferral circuit | Every app worker; inserted into `StdIn.deferrals` before execution |
| `DeferralInput` | Bincode, one per deferral circuit | Only the worker that produces the final internal proof (the tail merge) |

Whichever transport is used, the worker always reads its main input from the
deterministic path `/dev/shm/edge_{proof_uuid}/input.bin`.

### Flow 2 — manager-staged (default, `input_already_uploaded = false`)

The general path, and the only one that supports deferral and multi-element
inputs:

1. The caller uploads the bincode `StdIn` to the manager
   in a single `multipart/form-data` `POST /upload_input/{proof_uuid}` — the
   `input` part, plus (for a deferral proof) a `deferral_state_{i}` /
   `deferral_input_{i}` part per circuit. One upload call regardless of circuit
   count. The manager holds these in memory keyed by `proof_uuid`.
2. The caller calls `/start_proof` with `input_already_uploaded = false`.
3. The manager infers the deferral circuit count from the staged artifacts (a
   contiguous `0..N` set of `DeferralState`/`DeferralInput` pairs; `N = 0` is a
   plain non-deferral proof), fans the main input and each `DeferralState` out to
   every worker, and **retains** each `DeferralInput`.
4. Each `DeferralInput` is pushed **just-in-time** to the single worker the
   manager assigns the final internal prove, right before that dispatch — so the
   file is present when that worker runs the tail merge. It is never broadcast.

Because the bytes are uploaded to the manager (not referenced by a path), Flow 2
needs no shared filesystem between caller, manager, and workers.

### Flow 1 — worker pre-uploaded (`input_already_uploaded = true`)

The fast path for a high-throughput producer co-located with the workers: the
caller pushes the input directly to every worker before `/start_proof`, and the
manager skips fan-out. Deferral is **not** supported here (deferral artifacts
must be staged on the manager). Worker upload endpoints:

- `POST /upload_input/{proof_uuid}` accepts `bincode(StdIn<F>)` bytes.
- `POST /upload_input_compact/{proof_uuid}` accepts the raw bytes for one
  logical input element (smaller on the wire). The worker calls
  `StdIn::write_bytes`, serializes the resulting `StdIn`, and stores it as
  `input.bin`.

A direct-upload producer typically gets the registered worker URLs from the
manager's `/readyz`, uploads to every worker, then calls `/start_proof` with
`input_already_uploaded = true`.

### Transport is per-proof, not per-artifact

A single proof uses one transport for its main input; deferral always rides
Flow 2 (manager-staged), independent of how any Flow-1 producer would deliver a
non-deferral input. `DeferralInput` transport is decoupled from the main input:
the manager owns it end-to-end and delivers it only where it is consumed.

## Status

Manager proof status is one of:

- `in_progress`
- `failing` (transient: a worker reported a fatal error; the manager is draining peers before settling into `failed`)
- `completed`
- `failed`
- `canceled`

Terminal proof state is retained temporarily for status queries and then evicted.
