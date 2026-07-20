# Axiom Edge Architecture

## Components

```text
Client
  |
  | JSON API
  v
Edge Manager
  |       |
  |       | bincode result payloads
  |       v
  |   Proof State
  |
  | JSON and bincode work requests
  v
Edge Workers
  |
  v
OpenVM Provers
```

## Manager

The manager owns control-plane state:

- worker registry
- one active proof at a time
- proof status and recursion tree
- scheduler state
- result idempotency by recursion-tree slot (a duplicate result for an
  already-filled App/Leaf/Internal slot is detected and dropped)
- optional final proof persistence
- optional proof-lifecycle webhook (generic queued/proving/completed events)
- metrics emission

The manager pushes work to workers. Workers do not poll for work.

## Worker

Each worker owns a thread-pinned prover pool. Per-program CPU execution artifacts are built once per program, at startup for a seeded loadout and at registration otherwise, and shared across the pool; the GPU prover is built on idle provers when a registered program is published and swapped lazily when a job targets a different program, so a worker can serve any program in the loadout without holding every program's GPU state at once.

Worker endpoints:

- `/healthz`: process health
- `/readyz`: artifact and prover readiness, plus the programs this worker serves
- `/register_program`: derive a program's artifacts from a guest ELF and a VM config
- `/upload_input`: store bincode `StdIn` input
- `/upload_input_compact`: convert compact bytes to bincode `StdIn`
- `/sharded_app_prove`: start app proving for this worker's segments
- `/recursion_prove`: run leaf or internal aggregation work

## Data Flow

1. Worker starts, loads artifacts for every program `EDGE_PROGRAMS` seeds, creates its prover pool, and registers with the manager, which pushes every registered program the worker's `loaded_programs` omits.
2. Client calls `POST /start_proof`, naming the target `program` (`{name, version}`; optional when only one program is loaded).
3. Manager validates the program against its loadout, checks ready workers, and creates proof state.
4. Manager uploads input to workers unless the request says input is already uploaded.
5. Manager sends `/sharded_app_prove` to every worker (carrying the target program).
6. Workers stream `ExecuteE2`, app, leaf, internal, or error results to manager `/proof_result`.
7. Manager schedules follow-up leaf and internal work via `/recursion_prove`.
8. Manager marks the proof terminal when the final internal proof is present.

## State Model

Axiom Edge currently uses single-proof mode. A second proof is rejected while scheduler state exists for another proof.

This simplifies GPU memory management, but it means stuck proofs must reach a terminal state or be canceled before new proofs can start.

## Trust Boundary

### Deployment requirement: trusted network

The `edge-manager` and `edge-worker` HTTP binaries are **by design** plain
HTTP with no in-process auth, no body-size ceiling on large-payload routes,
and a fully-permissive CORS policy. They MUST be deployed on a trusted
private network and never be directly reachable from the public internet.
Any operator exposing them outside such a network is responsible for placing
an **external reverse proxy or load balancer** in front (e.g. Caddy, nginx,
Envoy, or a cloud load balancer) that terminates TLS and enforces
authentication, request-body limits, and a CORS policy.

Properties that hold by design (NOT bugs to be fixed inside these crates):

- manager endpoints are unauthenticated
- worker endpoints are unauthenticated
- worker registration trusts supplied URLs
- large worker payload routes (`/upload_input`, `/upload_input_compact`,
  `/recursion_prove`) call `DefaultBodyLimit::disable()` and `body.to_vec()`
  buffers the full upload before any application-level check, so a misbehaving
  client on the same network can exhaust worker memory; the body ceiling
  belongs at the proxy / LB.
- worker CORS is `allow_origin(Any).allow_methods(Any).allow_headers(Any)`

Rationale: keeping the binaries free of deployment-specific cross-cutting
security concerns lets the security front (reverse proxy / load balancer)
live entirely in deployment configuration and be chosen or swapped per
environment without rewiring application code.

## Failure Model

Expected terminal failures:

- app proving error
- leaf aggregation error
- internal aggregation error
- cost or segment validation failure
- caller cancellation

The configured proof timeout is enforced: a watchdog marks any proof exceeding `[proof] timeout_secs` as `failing` (then `failed`) and frees the scheduler slot.

### Worker prover-thread death

A worker builds its leaf/internal provers once at startup, inside each prover thread. If construction fails (or a thread later dies), that thread exits and its slot stays un-initialized — the worker is **not** restarted in-process, and there is no manager→worker cancel channel. This is an availability concern only: a dead thread yields no result, never a wrong one. Containment is external:

- The proof timeout above fails any proof that wedges on a dead worker (bounded by `[proof] timeout_secs`, not infinite).
- `/readyz` reports the node not-ready while any prover is un-initialized, so a load balancer gating on readiness stops routing to it.

Because recovery is not automatic, operators **must** alert on proof-failure rate and worker readiness, and restart a worker whose prover threads have died. Until then a node with a dead thread keeps failing each proof routed to it after the full timeout.
