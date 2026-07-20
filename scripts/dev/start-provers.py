#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2>=3.1"]
# ///
"""Dynamically generate TOML configs and docker-compose override for N GPU workers.

In --dry-run mode, prints the generated manager.toml, worker-N.toml, and
docker-compose.provers.yml to stdout, followed by the docker compose
command that would run.

Templates live in config/templates/ (jinja2). Edit those to change what's
written. Pass --extra-compose-file PATH (repeatable) to layer additional
docker-compose overlays on top of the generated stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DOCKER_DIR = REPO_ROOT / "docker"
TEMPLATES_DIR = REPO_ROOT / "config" / "templates"
DEFAULTS_FILE = REPO_ROOT / "config" / "defaults.toml"


def load_defaults() -> dict:
    """Load checked-in default config values from config/defaults.toml.

    This file is the single source of truth for defaults. Anything not
    overridden via CLI flags is read from here.
    """
    with DEFAULTS_FILE.open("rb") as f:
        return tomllib.load(f)


@dataclass
class Args:
    # Per-deploy values (no defaults.toml entries — inherently deploy-specific).
    num_gpus: int
    total_provers: int
    id_offset: int
    worker_only: bool
    manager_url_override: str
    worker_host: str
    worker_port_base: int
    no_build: bool
    dry_run: bool
    extra_compose_files: list[str]
    force_regenerate: bool
    features_override: str
    toolchain_override: str
    cpuset_enabled: bool
    cpuset_ranges_override: str
    cuda_arch_override: str
    persist_final_proofs_dir: str
    openvm_config_file: str
    compress_persisted_final_proofs: bool
    persist_leaf_failure_app_proofs_dir: str
    webhook_url: str
    # Resolved config values: CLI flag if given, otherwise from defaults.toml.
    app_provers: int
    leaf_provers: int
    internal_provers: int
    segment_memory: int | None
    leaf_pack_threshold: int
    timeout_secs: int
    leaf_arity: int
    internal_arity: int
    vpmm_page_size: int
    vpmm_pages: int
    metrics_endpoint: str
    metrics_output_dir: str
    manager_listen_addr: str
    worker_listen_addr: str
    log_level: str
    artifacts_path: str
    # Host directory holding the >10GB halo2 proving key + SRS files
    # (produced by the offline `halo2-keygen` binary). Mounted read-only
    # into worker containers at /data/halo2_pk when set; rendered into
    # `[artifacts] halo2_pk_path` in worker.toml. Empty string = no EVM
    # support (worker stays stark-only).
    halo2_pk_path: str
    metrics_output_path: str
    # Program loadout: list of dicts {"name": str, "version": int, "path": str}.
    programs: list[dict]
    # halo2 (EVM-wrap) proving mode: "none" | "full" | "dedicated". The two
    # booleans below are derived from it in `parse_args` and drive the rest of
    # the rendering; keeping them keeps the downstream code mode-agnostic.
    halo2_mode: str
    # Whether to add `evm-prove` to the build features (root -> halo2). True for
    # halo2 modes "full" and "dedicated"; false for "none" (stark-only). `--features`
    # *replaces* the whole default set (`cuda,jemalloc,parallel,aot,unprotected`),
    # so this instead appends evm-prove to the defaults.
    with_evm: bool
    # Deferral keyset wiring — independent of halo2. When `with_deferral=True`, run
    # `keygen --with-deferral` so `<artifacts>/deferral/cached_pk` is on disk,
    # and render `enable_deferral = true` into every worker.toml so the worker
    # derives that path and reconstructs the deferral-enabled SDK at boot. For the
    # EVM deferral flow (proof_type=evm) also select a halo2 mode that builds
    # evm-prove (full/dedicated). Deferral itself does not require evm-prove.
    with_deferral: bool
    # halo2 "dedicated" mode (derived: halo2_mode == "dedicated"). The highest-id
    # worker (prover_id == total_provers-1) renders `worker_role = evm_dedicated`
    # and runs only root -> halo2; the remaining workers render `worker_role =    # normal`. The halo2 key is rendered into ONLY the dedicated worker's
    # worker.toml. In "none"/"full" no `worker_role` is rendered (serde default
    # Full) and (in "full") the halo2 key mounts on every worker.
    dedicated_halo2_gpu: bool


def parse_args(defaults: dict) -> Args:
    p = argparse.ArgumentParser(
        prog="start-provers.py",
        description="Generate TOML configs and docker-compose override for N GPU workers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("num_gpus", nargs="?", type=int, default=None, help="GPUs on this machine (default: 4)")
    p.add_argument("--total-provers", type=int, default=None, metavar="N",
                   help="Total workers across all machines (default: num_gpus)")
    p.add_argument("--id-offset", type=int, default=0, metavar="N",
                   help="Starting prover_id for this machine (default: 0)")
    p.add_argument("--worker-only", action="store_true",
                   help="Skip manager, only start GPU workers")
    p.add_argument("--manager-url", dest="manager_url_override", default="", metavar="URL",
                   help="Remote manager URL (required with --worker-only)")
    p.add_argument("--worker-host", default="", metavar="IP",
                   help="Override auto-detected IP for worker registration")
    p.add_argument("--worker-port-base", type=int, default=8001, metavar="PORT",
                   help="Base host port for workers; worker i publishes PORT+i "
                        "(default: 8001). Bump this if the host reserves the "
                        "8001+ range (e.g. a forwarded SSH port).")
    p.add_argument("--no-build", action="store_true",
                   help="Skip Docker image build")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate configs + compose override, print to stdout, skip build/up")
    p.add_argument("--extra-compose-file", dest="extra_compose_files", action="append",
                   default=[], metavar="PATH",
                   help="Additional docker-compose overlay file, relative to repo root. Repeatable.")
    p.add_argument(
        "--regenerate",
        "--regenerate-artifacts",
        dest="force_regenerate",
        action="store_true",
        help="Force regenerate proving artifacts before startup",
    )
    p.add_argument("--features", dest="features_override", default="", metavar="LIST",
                   help="Cargo features for edge-worker image build (default: [build].features in defaults.toml)")
    p.add_argument("--toolchain", dest="toolchain_override", default="", metavar="TOOLCHAIN",
                   help="Rust toolchain for image build (default: [build].toolchain in defaults.toml)")
    p.add_argument("--no-cpuset", dest="cpuset_enabled", action="store_false",
                   help="Disable automatic per-worker CPU pinning")
    p.add_argument("--cpuset-ranges", dest="cpuset_ranges_override", default="", metavar="RANGES",
                   help="Comma-separated cpuset list per local worker (e.g. '0-15,16-31,32-47,48-63')")
    # Defaults for the flags below live in config/defaults.toml; passing the
    # flag overrides the value in that file for this run.
    p.add_argument("--app-provers", type=int, default=None, metavar="N",
                   help="App prover instances per worker service (default: from defaults.toml)")
    p.add_argument("--leaf-provers", type=int, default=None, metavar="N",
                   help="Leaf prover instances per worker service (default: from defaults.toml)")
    p.add_argument("--internal-provers", type=int, default=None, metavar="N",
                   help="Internal prover instances per worker service (default: from defaults.toml)")
    p.add_argument("--segment-memory", type=int, default=None, metavar="N",
                   help="OpenVM max segment memory override (default: from defaults.toml; unset = OpenVM default)")
    p.add_argument("--vpmm-page-size", type=int, default=None, metavar="N",
                   help="CUDA VPMM page size override (default: from defaults.toml)")
    p.add_argument("--vpmm-pages", type=int, default=None, metavar="N",
                   help="CUDA VPMM page count override (default: from defaults.toml)")
    p.add_argument("--metrics-endpoint", default=None, metavar="URL",
                   help="OTLP metrics endpoint override (default: from defaults.toml)")
    p.add_argument("--metrics-output-dir", default=None, metavar="DIR",
                   help="Manager metrics output directory override (default: from defaults.toml)")
    p.add_argument("--leaf-arity", type=int, default=None, metavar="N",
                   help="Leaf-circuit fan-in: app proofs aggregated per leaf proof (default: from defaults.toml)")
    p.add_argument("--internal-arity", type=int, default=None, metavar="N",
                   help="Internal-circuit fan-in: child proofs aggregated per internal proof (default: from defaults.toml)")
    p.add_argument("--cuda-arch", dest="cuda_arch_override", default="", metavar="ARCH",
                   help="CUDA arch list for image build (default: [build].cuda_arch in defaults.toml, e.g. 89 or 89,120)")
    p.add_argument("--openvm-config-file", default="", metavar="TOML",
                   help="Custom OpenVM app_vm_config TOML (extension set etc.) used for "
                        "keygen + ELF→vmexe. Default: the built-in standard config. "
                        "Changing it forces key regeneration.")
    p.add_argument("--halo2-pk-path", dest="halo2_pk_path", default="", metavar="DIR",
                   help="Host directory holding the halo2 proving key + SRS files "
                        "(produced by the offline `halo2-keygen` binary). Mounted "
                        "read-only into worker containers at /data/halo2_pk and "
                        "rendered into [artifacts] halo2_pk_path in worker.toml. "
                        "Required for proof_type=evm; leave unset for stark-only "
                        "deployments.")
    p.add_argument("--halo2", dest="halo2", choices=["none", "full", "dedicated"],
                   default="none",
                   help="halo2 (EVM-wrap) proving mode. "
                        "'none' (default): stark-only — no halo2 at all; the "
                        "`evm-prove` build feature is NOT added and no halo2 key is "
                        "needed (serves proof_type=stark only). "
                        "'full': every worker is eligible for the "
                        "manager-dispatched root -> halo2 EVM step (adds "
                        "`evm-prove` to the default build features; "
                        "needs --halo2-pk-path). "
                        "'dedicated': isolate halo2 on the highest-id worker "
                        "(runs root -> halo2 only, on its own GPU) while the rest "
                        "run app/leaf/internal; the halo2 key mounts on that worker "
                        "only (adds `evm-prove`; needs --halo2-pk-path and >= 2 "
                        "workers). Deferral is an independent toggle "
                        "(--with-deferral) and composes with any halo2 mode.")
    p.add_argument("--with-deferral", dest="with_deferral", action="store_true",
                   help="Deploy with deferral support (verify-stark): run "
                        "`keygen --with-deferral` so workers can reconstruct a "
                        "deferral-enabled SDK from `<artifacts>/deferral/cached_pk` "
                        "and exercise the tail merge (`prove_def → prove_mixed → "
                        "wrap`). For the EVM deferral flow (proof_type=evm) also "
                        "set `--halo2 full` (or `--halo2 dedicated`). Off by default.")
    p.add_argument("--persist-final-proofs-dir", default="", metavar="DIR",
                   help="Ask the manager to persist completed final proofs as bincode files")
    p.add_argument("--compress-persisted-final-proofs", action="store_true",
                   help="When persisting final proofs, zstd-compress the .proof.bin payload before writing")
    p.add_argument("--persist-leaf-failure-app-proofs-dir", default="", metavar="DIR",
                   help="Ask the manager to snapshot app proofs when leaf logup nonzero-root-sum failures occur")
    p.add_argument("--webhook-url", default="", metavar="URL",
                   help="Generic proof-lifecycle webhook URL. When set, the manager "
                        "POSTs queued/proving/completed events here for an external "
                        "consumer (e.g. a reporter sidecar) to translate.")
    p.add_argument(
        "--programs", dest="programs_input", default="", metavar="JSON_OR_PATH",
        required=True,
        help=(
            "Required. Program loadout as JSON list of {name, version, path} "
            "objects, or a path to a JSON file with the same shape."
        ),
    )

    ns = p.parse_args()

    num_gpus = ns.num_gpus if ns.num_gpus is not None else 4
    total_provers = ns.total_provers if ns.total_provers is not None else num_gpus

    programs = resolve_programs(ns.programs_input)

    def required(section: str, key: str):
        try:
            return defaults[section][key]
        except KeyError:
            sys.stderr.write(
                f"config/defaults.toml is missing required key [{section}].{key}\n"
            )
            sys.exit(1)

    def optional(section: str, key: str):
        return defaults.get(section, {}).get(key)

    def pick(cli_val, section: str, key: str):
        return cli_val if cli_val is not None else required(section, key)

    def pick_optional(cli_val, section: str, key: str):
        # Optional values (e.g. segment overrides) where None = "don't render
        # the key at all". CLI takes precedence; otherwise fall back to
        # defaults.toml; otherwise None.
        if cli_val is not None:
            return cli_val
        return optional(section, key)

    return Args(
        num_gpus=num_gpus,
        total_provers=total_provers,
        id_offset=ns.id_offset,
        worker_only=ns.worker_only,
        manager_url_override=ns.manager_url_override,
        worker_host=ns.worker_host,
        worker_port_base=ns.worker_port_base,
        no_build=ns.no_build,
        dry_run=ns.dry_run,
        extra_compose_files=list(ns.extra_compose_files),
        force_regenerate=ns.force_regenerate,
        features_override=ns.features_override,
        toolchain_override=ns.toolchain_override,
        cpuset_enabled=ns.cpuset_enabled,
        cpuset_ranges_override=ns.cpuset_ranges_override,
        cuda_arch_override=ns.cuda_arch_override,
        persist_final_proofs_dir=ns.persist_final_proofs_dir,
        compress_persisted_final_proofs=ns.compress_persisted_final_proofs,
        persist_leaf_failure_app_proofs_dir=ns.persist_leaf_failure_app_proofs_dir,
        webhook_url=ns.webhook_url,
        # Absolute path so the generation binaries resolve it regardless of cwd.
        openvm_config_file=(
            str(Path(ns.openvm_config_file).expanduser().resolve())
            if ns.openvm_config_file
            else ""
        ),
        # Resolved from CLI > defaults.toml.
        app_provers=pick(ns.app_provers, "provers", "max_app_provers"),
        leaf_provers=pick(ns.leaf_provers, "provers", "max_leaf_provers"),
        internal_provers=pick(ns.internal_provers, "provers", "max_internal_provers"),
        segment_memory=pick_optional(ns.segment_memory, "provers", "default_segment_memory"),
        leaf_pack_threshold=required("proof", "leaf_pack_threshold"),
        timeout_secs=required("proof", "timeout_secs"),
        leaf_arity=pick(ns.leaf_arity, "proof", "leaf_arity"),
        internal_arity=pick(ns.internal_arity, "proof", "internal_arity"),
        vpmm_page_size=pick(ns.vpmm_page_size, "cuda", "vpmm_page_size"),
        vpmm_pages=pick(ns.vpmm_pages, "cuda", "vpmm_pages"),
        metrics_endpoint=pick(ns.metrics_endpoint, "metrics", "endpoint"),
        metrics_output_dir=pick(ns.metrics_output_dir, "metrics", "output_dir"),
        manager_listen_addr=required("server", "manager_listen_addr"),
        worker_listen_addr=required("server", "worker_listen_addr"),
        log_level=required("telemetry", "log_level"),
        artifacts_path=required("paths", "artifacts_path"),
        # halo2_pk_path: optional, CLI-only. defaults.toml has no entry —
        # stark-only deployments leave it unset.
        halo2_pk_path=(
            str(Path(ns.halo2_pk_path).expanduser().resolve())
            if ns.halo2_pk_path
            else ""
        ),
        metrics_output_path=required("paths", "metrics_output_path"),
        programs=programs,
        # halo2 mode drives the two booleans below: full/dedicated build
        # evm-prove; dedicated additionally isolates halo2 on the top-id worker.
        halo2_mode=ns.halo2,
        with_evm=ns.halo2 in ("full", "dedicated"),
        with_deferral=ns.with_deferral,
        dedicated_halo2_gpu=ns.halo2 == "dedicated",
    )


# --- Program loadout resolution ---


def resolve_programs(programs_input: str) -> list[dict]:
    """Resolve the program loadout from --programs.

    `--programs` is required: either inline JSON (a list of
    {name, version, path} objects) or a path to a JSON file with the same
    shape. No environment-variable fallback.

    Returns a list of {"name": str, "version": int, "path": str} dicts.
    """
    raw = programs_input.strip()
    if not raw:
        sys.stderr.write(
            "--programs is required: pass a JSON list of {name, version, path} "
            "objects, or a path to a JSON file with the same shape\n"
        )
        sys.exit(1)

    # Could be a path or inline JSON.
    if not raw.startswith("["):
        path = Path(raw)
        if not path.is_file():
            sys.stderr.write(
                f"--programs value {raw!r} is neither inline JSON nor an existing file\n"
            )
            sys.exit(1)
        raw = path.read_text()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Failed to parse programs JSON: {e}\n")
        sys.exit(1)

    if not isinstance(parsed, list) or not parsed:
        sys.stderr.write(
            "programs JSON must be a non-empty array of {name, version, path} objects\n"
        )
        sys.exit(1)

    seen = set()
    normalized: list[dict] = []
    for i, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            sys.stderr.write(f"programs entry {i} is not an object\n")
            sys.exit(1)
        for key in ("name", "version", "path"):
            if key not in entry:
                sys.stderr.write(f"programs entry {i} missing required key {key!r}\n")
                sys.exit(1)
        name = str(entry["name"])
        try:
            version = int(entry["version"])
        except (TypeError, ValueError):
            sys.stderr.write(f"programs entry {i} 'version' must be an integer\n")
            sys.exit(1)
        # Expand ~ in paths so JSON files can be portable across hosts.
        path = os.path.expanduser(str(entry["path"]))
        key = (name, version)
        if key in seen:
            sys.stderr.write(f"programs has duplicate entry for ({name}, v{version})\n")
            sys.exit(1)
        seen.add(key)
        normalized.append({"name": name, "version": version, "path": path})

    return normalized


def validate_program_paths(programs: list[dict]) -> None:
    for program in programs:
        name = program["name"]
        version = program["version"]
        path = program["path"]
        if not Path(path).is_file():
            sys.stderr.write(
                f"programs entry ({name}, v{version}): ELF not found at {path}\n"
            )
            sys.exit(1)


def programs_env_value(programs: list[dict]) -> str:
    """Render the EDGE_PROGRAMS env value seen by manager + workers.

    Includes only {name, version} — `path` is start-provers' input, not
    something the Rust binaries care about.
    """
    return json.dumps([{"name": p["name"], "version": p["version"]} for p in programs])


def validate(args: Args) -> None:
    def die(msg: str) -> None:
        sys.stderr.write(msg + "\n")
        sys.exit(1)

    if args.num_gpus <= 0:
        die("num_gpus must be a positive integer")
    if args.id_offset < 0:
        die("--id-offset must be a non-negative integer")
    if args.total_provers <= 0:
        die("--total-provers must be a positive integer")
    for extra in args.extra_compose_files:
        if not (REPO_ROOT / extra).is_file():
            die(f"--extra-compose-file path does not exist: {extra}")
    if args.app_provers <= 0:
        die("--app-provers must be a positive integer")
    if args.leaf_provers <= 0:
        die("--leaf-provers must be a positive integer")
    if args.cuda_arch_override and not re.fullmatch(
        r"[0-9]+(,[0-9]+)*", args.cuda_arch_override
    ):
        die(
            "--cuda-arch must be a comma-separated list of numeric arch values "
            "(for example 89 or 89,120)"
        )
    if args.worker_only and not args.manager_url_override:
        die("--manager-url is required when using --worker-only")
    if args.worker_only and args.persist_final_proofs_dir:
        die(
            "--persist-final-proofs-dir cannot be used with --worker-only "
            "because no manager is started"
        )
    if args.worker_only and args.persist_leaf_failure_app_proofs_dir:
        die(
            "--persist-leaf-failure-app-proofs-dir cannot be used with --worker-only "
            "because no manager is started"
        )
    if args.worker_only and args.compress_persisted_final_proofs:
        die(
            "--compress-persisted-final-proofs cannot be used with --worker-only "
            "because no manager is started"
        )
    if args.compress_persisted_final_proofs and not args.persist_final_proofs_dir:
        die("--compress-persisted-final-proofs requires --persist-final-proofs-dir")
    if args.worker_only and args.webhook_url:
        die("--webhook-url cannot be used with --worker-only (no manager is started)")
    if args.openvm_config_file and not Path(args.openvm_config_file).is_file():
        die(f"--openvm-config-file not found: {args.openvm_config_file}")
    if not args.cpuset_enabled and args.cpuset_ranges_override:
        die("--cpuset-ranges cannot be used with --no-cpuset")
    if args.with_deferral and not args.with_evm:
        # Deferral proving itself no longer requires evm-prove (stark-mode
        # deferral works without root/halo2). But the EVM deferral flow
        # (proof_type=evm) does, and that's the common intent. If the caller
        # passed an explicit --features list that omits evm-prove and didn't
        # add --with-evm, flag it — cheaper than a runtime failure when an
        # evm-typed deferral job hits the missing root/halo2 machinery. (Pass
        # --with-evm to add evm-prove, or drop this check by running a genuine
        # stark-only deferral deployment with the default features.)
        feature_list = [f.strip() for f in args.features_override.split(",") if f.strip()]
        if feature_list and "evm-prove" not in feature_list:
            die(
                "--with-deferral with an explicit --features list that omits "
                "`evm-prove`: EVM deferral jobs (proof_type=evm) need it. Select "
                "--halo2 full (or dedicated) (recommended) or put evm-prove in "
                "--features. For a stark-only deferral deployment, run without an "
                "explicit --features override."
            )
    if args.dedicated_halo2_gpu:
        # The dedicated worker is the GLOBAL highest-id worker
        # (prover_id == total_provers-1); the rest are stark-only (ids 0..N-2). At
        # least one stark-only worker must remain to run app/leaf/internal, else the
        # deployment can prove nothing — so the pool needs >= 2 total workers.
        if args.total_provers < 2:
            die(
                "--halo2 dedicated needs at least 2 total workers: the "
                "highest-id worker becomes the halo2-only dedicated worker, so "
                "there must be at least one other worker (ids 0..N-2) left to "
                "run app/leaf/internal. Got --total-provers="
                f"{args.total_provers}."
            )
        # Only the machine that actually hosts the top id needs the halo2 key and
        # the evm-prove feature. (In a single-machine deploy — id_offset 0,
        # num_gpus == total_provers — that's always this machine.) Guard those
        # requirements just for that machine so a stark-only satellite machine in
        # a multi-host dedicated deploy can still pass --halo2 dedicated to get
        # its `worker_role = stark_only` render.
        top_id = args.total_provers - 1
        machine_hosts_dedicated = (
            args.id_offset <= top_id < args.id_offset + args.num_gpus
        )
        if machine_hosts_dedicated:
            if not args.halo2_pk_path:
                die(
                    "--halo2 dedicated designates worker "
                    f"{top_id} on this machine as the halo2-only dedicated "
                    "worker, but --halo2-pk-path was not given, so it would have "
                    "no halo2 key to load. Pass --halo2-pk-path."
                )
            # The dedicated worker builds the root + halo2 provers, which only
            # exist in an evm-prove build. --with-evm appends evm-prove; an
            # explicit --features must include it. Mirrors the --with-deferral
            # check above so the failure is caught at deploy time, not at boot.
            # `--halo2 dedicated` sets with_evm (evm-prove appended), so this is
            # only reachable if an explicit --features override REPLACED the set
            # without evm-prove — the dedicated worker couldn't build root/halo2.
            feature_list = [f.strip() for f in args.features_override.split(",") if f.strip()]
            if feature_list and "evm-prove" not in feature_list:
                die(
                    "--halo2 dedicated needs the evm-prove build feature so the "
                    "dedicated worker can build the root + halo2 provers, but your "
                    "explicit --features override omits it. Include evm-prove in "
                    "--features (or drop the override)."
                )


def compute_cpuset_ranges(args: Args) -> list[str]:
    if not args.cpuset_enabled:
        return []
    if args.cpuset_ranges_override:
        parts = args.cpuset_ranges_override.split(",")
        if len(parts) != args.num_gpus:
            sys.stderr.write(
                f"--cpuset-ranges must provide exactly {args.num_gpus} entries "
                f"(got {len(parts)})\n"
            )
            sys.exit(1)
        for part in parts:
            if not re.fullmatch(r"[0-9,-]+", part):
                sys.stderr.write(f"Invalid cpuset range '{part}' in --cpuset-ranges\n")
                sys.exit(1)
        return parts
    host_cpus = os.cpu_count() or 0
    if host_cpus <= 0:
        sys.stderr.write("Failed to detect host CPU count for cpuset generation\n")
        sys.exit(1)
    if host_cpus < args.num_gpus:
        sys.stderr.write(
            f"Host CPU count ({host_cpus}) is smaller than num_gpus ({args.num_gpus}); "
            "cannot auto-generate cpuset ranges\n"
        )
        sys.exit(1)
    ranges = []
    for i in range(args.num_gpus):
        start = i * host_cpus // args.num_gpus
        end = ((i + 1) * host_cpus // args.num_gpus) - 1
        if end < start:
            end = start
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
    return ranges


def auto_detect_worker_host() -> str:
    """Mirror bash: hostname -I | awk '{print $1}' || ip route get 1 ..."""
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, check=False
        ).stdout.strip()
        first = out.split()[0] if out else ""
        if first:
            return first
    except FileNotFoundError:
        pass
    try:
        out = subprocess.run(
            ["ip", "route", "get", "1"], capture_output=True, text=True, check=False
        ).stdout
        m = re.search(r"\bsrc\s+(\S+)", out)
        if m:
            return m.group(1)
        first_line = out.splitlines()[0] if out else ""
        toks = first_line.split()
        if len(toks) >= 7:
            return toks[6]
    except FileNotFoundError:
        pass
    return ""


def resolve_effective(args: Args, defaults: dict) -> dict[str, str]:
    """Resolve build values from `[build]` in defaults.toml, applying CLI
    overrides where flags exist.

    No environment variables are consulted for config — defaults.toml is the
    single source of truth, CLI flags override per-run.
    """
    build = defaults.get("build", {})

    def build_required(key: str) -> str:
        try:
            return str(build[key])
        except KeyError:
            sys.stderr.write(
                f"config/defaults.toml is missing required key [build].{key}\n"
            )
            sys.exit(1)

    features = args.features_override or build_required("features")
    if args.with_evm:
        # Append (not replace) evm-prove so the default feature set is kept.
        feat_list = [f.strip() for f in features.split(",") if f.strip()]
        if "evm-prove" not in feat_list:
            feat_list.append("evm-prove")
        features = ",".join(feat_list)
    toolchain = args.toolchain_override or build_required("toolchain")

    if args.cuda_arch_override:
        cuda_arch = args.cuda_arch_override
        cuda_arch_source = "flag"
    else:
        cuda_arch = build_required("cuda_arch")
        cuda_arch_source = "defaults.toml"

    manager_url = args.manager_url_override or "http://edge-manager:3000"

    return {
        "features": features,
        "toolchain": toolchain,
        "cuda_arch": cuda_arch,
        "cuda_arch_source": cuda_arch_source,
        "jemalloc": build_required("jemalloc"),
        "manager_url": manager_url,
        # Docker build args (previously env-driven; now defaults.toml only).
        "cargo_build_jobs": build_required("cargo_build_jobs"),
        "profile": build_required("profile"),
        "target_cpu": build_required("target_cpu"),
        "builder_image": build_required("builder_image"),
        "runtime_image": build_required("runtime_image"),
    }


# --- Revision tracking (git + Cargo.lock) ---


def git_rev_parse_head(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return ""


def get_lock_rev(name: str, cargo_lock: Path) -> str:
    """Find git rev hash for a crate in Cargo.lock."""
    try:
        content = cargo_lock.read_text()
    except FileNotFoundError:
        return ""
    in_block = False
    for line in content.splitlines():
        if not in_block:
            if line.strip() == f'name = "{name}"':
                in_block = True
            continue
        if line.strip() == "":
            break
        m = re.search(r'#([0-9a-f]+)"', line)
        if m:
            return m.group(1)
    return ""


def compute_vcs_refs(script_dir: Path, build_workspace: Path) -> dict[str, str]:
    return {
        "vcs_ref": git_rev_parse_head(script_dir) or "unknown",
        "openvm_rev": get_lock_rev("openvm-sdk", build_workspace / "Cargo.lock") or "unknown",
        "stark_rev": get_lock_rev("openvm-stark-sdk", build_workspace / "Cargo.lock") or "unknown",
    }


# --- Banner / status output ---


def print_banner(
    args: Args,
    eff: dict[str, str],
    vcs_refs: dict[str, str],
    cpuset_ranges: list[str],
) -> None:
    print("=== Axiom Edge Prover Setup ===")
    print(f"  GPUs on this machine: {args.num_gpus}")
    print(f"  Total provers:        {args.total_provers}")
    print(f"  ID offset:            {args.id_offset}")
    print(f"  Worker-only:          {str(args.worker_only).lower()}")
    print(f"  Manager URL:          {eff['manager_url']}")
    for extra in args.extra_compose_files:
        print(f"  Extra compose file:   {extra}")
    print(f"  Force regenerate:     {str(args.force_regenerate).lower()}")
    print(f"  App provers/worker:   {args.app_provers}")
    print(f"  halo2 mode:           {args.halo2_mode}")
    print(f"  Deferral:             {str(args.with_deferral).lower()}")
    print(f"  Build features:       {eff['features']}")
    print(f"  Build toolchain:      {eff['toolchain'] or 'default'}")
    print(f"  CUDA arch:            {eff['cuda_arch']} ({eff['cuda_arch_source']})")
    print(f"  Repo revision:        {vcs_refs['vcs_ref']}")
    print(f"  OpenVM revision:      {vcs_refs['openvm_rev']}")
    print(f"  Stark revision:       {vcs_refs['stark_rev']}")
    if args.worker_host:
        print(f"  Worker host:          {args.worker_host}")
    if args.no_build:
        print(
            "  NOTE: --no-build keeps existing images; feature/toolchain changes apply only after rebuild."
        )
    if args.cpuset_enabled:
        print("  CPU pinning:          enabled")
        for i in range(args.num_gpus):
            print(f"    edge-worker-{i} -> cpuset {cpuset_ranges[i]}")
    else:
        print("  CPU pinning:          disabled")
    if args.openvm_config_file:
        print(f"  OpenVM config:        {args.openvm_config_file}")
    else:
        print("  OpenVM config:        standard (built-in)")
    if args.persist_final_proofs_dir:
        print(f"  Final proof dir:      {args.persist_final_proofs_dir}")
    if args.compress_persisted_final_proofs:
        print("  Final proof compress: enabled")
    if args.persist_leaf_failure_app_proofs_dir:
        print(f"  Leaf failure app dir: {args.persist_leaf_failure_app_proofs_dir}")
    if args.webhook_url:
        print(f"  Lifecycle webhook:    {args.webhook_url}")
    if args.with_deferral:
        print("  Deferral keyset:      enabled (verify-stark)")
    if args.dedicated_halo2_gpu:
        dedicated_id = args.total_provers - 1
        print(
            f"  Dedicated halo2 GPU:  enabled (worker {dedicated_id} = EvmDedicated; "
            f"{dedicated_id} stark-only worker(s))"
        )
    print(f"  Jemalloc conf:        {eff['jemalloc']}")
    print(f"  Programs:             {len(args.programs)}")
    for p in args.programs:
        print(f"    {p['name']} v{p['version']} <- {p['path']}")
    print()


def build_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def banner_for(template_name: str) -> str:
    """The banner prepended to every rendered file.

    Kept out of the template itself so the .j2 files don't pretend to be
    auto-generated.
    """
    return f"# AUTO-GENERATED — do not edit. Source: config/templates/{template_name}\n\n"


def render_manager_toml(env: Environment, args: Args) -> str:
    tpl = env.get_template("manager.toml.j2")
    return banner_for("manager.toml.j2") + tpl.render(
        manager_listen_addr=args.manager_listen_addr,
        num_workers=args.total_provers,
        max_app_provers=args.app_provers,
        max_leaf_provers=args.leaf_provers,
        max_internal_provers=args.internal_provers,
        leaf_pack_threshold=args.leaf_pack_threshold,
        timeout_secs=args.timeout_secs,
        leaf_arity=args.leaf_arity,
        internal_arity=args.internal_arity,
        persist_final_proofs_dir=args.persist_final_proofs_dir,
        compress_persisted_final_proofs=args.compress_persisted_final_proofs,
        persist_leaf_failure_app_proofs_dir=args.persist_leaf_failure_app_proofs_dir,
        lifecycle_webhook_url=args.webhook_url,
        metrics_output_dir=args.metrics_output_dir,
        metrics_endpoint=args.metrics_endpoint,
        log_level=args.log_level,
    )


def worker_role_for(args: Args, prover_id: int) -> str | None:
    """WorkerRole serde string for the worker with this `prover_id`, or None.

    Off (default): returns None for every worker → the `worker_role` field is
    omitted from worker.toml → serde defaults to `Full` → byte-identical to a
    render without the flag.

    On (`--dedicated-halo2-gpu`): the single GLOBAL highest-id worker
    (`prover_id == total_provers - 1`) is `evm_dedicated`; every other worker is
    `stark_only`. Pinning the dedicated role to the top id is what keeps the stark-only
    workers on contiguous ids `0..N-2` — the manager's app-sharding math
    assigns segment `s` to `s % num_provers` over that contiguous set, so a
    middle dedicated index would orphan a shard and stall the proof.
    """
    if not args.dedicated_halo2_gpu:
        return None
    if prover_id == args.total_provers - 1:
        return "evm_dedicated"
    return "stark_only"


def render_service_toml(env: Environment, args: Args, i: int, manager_url: str) -> str:
    tpl = env.get_template("worker.toml.j2")
    prover_id = args.id_offset + i
    if args.worker_host:
        worker_url = f"http://{args.worker_host}:{args.worker_port_base + i}"
    else:
        worker_url = f"http://edge-worker-{i}:8001"

    role = worker_role_for(args, prover_id)
    is_dedicated = role == "evm_dedicated"
    # halo2 key placement:
    #   - default (flag off): mount on every worker whenever --halo2-pk-path was
    #     passed — today's behavior, byte-identical.
    #   - dedicated mode: render the key into ONLY the dedicated worker's
    #     worker.toml so stark-only workers never load the ~10GB halo2 pk (the whole
    #     point of the isolation). A worker loads halo2 iff its worker.toml sets
    #     [artifacts] halo2_pk_path, so gating this line is what gates the load.
    if args.dedicated_halo2_gpu:
        render_halo2 = bool(args.halo2_pk_path) and is_dedicated
    else:
        render_halo2 = bool(args.halo2_pk_path)
    # The dedicated worker runs ONLY the EVM step (root + halo2) — no
    # app/leaf/internal work — so it must report ZERO app/leaf/internal capacity.
    # That is exactly what the manager's role-aware `validate_provers` requires
    # for the EvmDedicated role; reporting non-zero leaf/internal here gets the
    # worker rejected at registration. (Root's recursive prover is built from the
    # agg key, not the leaf/internal pools, so zero is correct.) Every other
    # worker keeps the configured counts.
    max_app_provers = 0 if is_dedicated else args.app_provers
    max_leaf_provers = 0 if is_dedicated else args.leaf_provers
    max_internal_provers = 0 if is_dedicated else args.internal_provers

    return banner_for("worker.toml.j2") + tpl.render(
        worker_listen_addr=args.worker_listen_addr,
        prover_id=prover_id,
        num_provers=args.total_provers,
        worker_url=worker_url,
        manager_url=manager_url,
        # Rendered only in dedicated mode; empty ⇒ the template omits the
        # `worker_role` line ⇒ serde default `Full` (today's behavior).
        worker_role=role or "",
        max_app_provers=max_app_provers,
        max_leaf_provers=max_leaf_provers,
        max_internal_provers=max_internal_provers,
        default_segment_memory=args.segment_memory if args.segment_memory is not None else "",
        # `halo2_pk_path` is rendered only when --halo2-pk-path was passed
        # (i.e. EVM-capable deploy). Container-side path is fixed; the host
        # dir is mounted into the worker at /data/halo2_pk by the compose
        # template, matching the ARTIFACTS_PATH pattern. In dedicated-halo2 mode
        # only the EvmDedicated worker renders it (see render_halo2 above).
        halo2_pk_path="/data/halo2_pk" if render_halo2 else "",
        # `enable_deferral` is a plain toggle (not a path): the deferral
        # cached_pk lives at the conventional `<artifacts>/deferral/cached_pk`
        # alongside the other shared keys, so the worker derives the location.
        # Rendered as `enable_deferral = true` only when --with-deferral was
        # passed (and `keygen --with-deferral` wrote the artifact).
        enable_deferral=args.with_deferral,
        vpmm_page_size=args.vpmm_page_size,
        vpmm_pages=args.vpmm_pages,
        log_level=args.log_level,
    )


def render_compose_override(
    env: Environment,
    args: Args,
    cpuset_ranges: list[str],
    jemalloc: str,
) -> str:
    tpl = env.get_template("docker-compose.provers.yml.j2")
    workers = [
        {
            "index": i,
            "http_port": args.worker_port_base + i,
            "cpuset": cpuset_ranges[i] if args.cpuset_enabled else "",
        }
        for i in range(args.num_gpus)
    ]
    return banner_for("docker-compose.provers.yml.j2") + tpl.render(
        worker_only=args.worker_only,
        cpuset_enabled=args.cpuset_enabled,
        persist_final_proofs_dir=args.persist_final_proofs_dir,
        persist_leaf_failure_app_proofs_dir=args.persist_leaf_failure_app_proofs_dir,
        # When set, the compose template adds a `${HALO2_PK_PATH}:/data/halo2_pk:ro`
        # mount to every worker — same shape as the ARTIFACTS_PATH mount.
        halo2_pk_path=args.halo2_pk_path,
        jemalloc=jemalloc,
        workers=workers,
        vpmm_page_size=args.vpmm_page_size,
        vpmm_pages=args.vpmm_pages,
    )


# --- Artifact keygen ---


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_artifacts(
    args: Args, features: str, toolchain: str, build_workspace: Path
) -> bool:
    """Generate shared keys + per-program vmexes if missing or forced.

    Disk layout:
        {artifacts_path}/app_pk                                # shared
        {artifacts_path}/agg_stark_pk                          # shared
        {artifacts_path}/programs/{name}/{version}/program.vmexe   # per-ELF

    Returns True if any artifact generation actually ran.
    """
    artifacts = Path(args.artifacts_path)
    convert_bin = build_workspace / "target/release/convert_fixtures"
    keygen_bin = build_workspace / "target/release/keygen"
    version_file = artifacts / ".keygen-inputs-hash"
    cargo_lock = build_workspace / "Cargo.lock"
    lock_hash = _sha256_file(cargo_lock) if cargo_lock.is_file() else "unknown"

    # Keys depend on both the dependency set (Cargo.lock) and the VM config. A
    # custom --openvm-config-file changes the baked-in config, so its content
    # must invalidate the cached keys too — otherwise a stale app_pk silently
    # mismatches the new config (mismatched-prime class of failure).
    openvm_config_hash = (
        _sha256_file(Path(args.openvm_config_file)) if args.openvm_config_file else "standard"
    )
    inputs_hash = hashlib.sha256(f"{lock_hash}:{openvm_config_hash}".encode()).hexdigest()

    # Under evm-prove the keygen bin also emits root_pk (cheap host-side, no
    # KZG). Require it for the freshness check so flipping a stark-only
    # deploy to evm-prove forces a regen even if Cargo.lock/openvm-config
    # didn't change (the inputs-hash gate alone wouldn't catch this).
    evm_prove_enabled = "evm-prove" in [f.strip() for f in features.split(",")]
    shared_keys_present = (artifacts / "app_pk").is_file() and (
        artifacts / "agg_stark_pk"
    ).is_file()
    if evm_prove_enabled:
        shared_keys_present = shared_keys_present and (artifacts / "root_pk").is_file()
    if args.with_deferral:
        # The deferral artifact `<artifacts>/deferral/cached_pk` is what the
        # worker loads at boot when `enable_deferral = true` is rendered.
        # Treat it the same way as `root_pk` for the freshness check: flipping
        # a deploy to --with-deferral must force a regen if the file is
        # missing, even when Cargo.lock + openvm-config didn't change.
        shared_keys_present = (
            shared_keys_present
            and (artifacts / "deferral" / "cached_pk").is_file()
        )
    keygen_inputs_stale = (
        not version_file.is_file() or version_file.read_text().strip() != inputs_hash
    )

    need_shared_keygen = args.force_regenerate or not shared_keys_present or keygen_inputs_stale
    if need_shared_keygen:
        if args.force_regenerate:
            print("Regeneration requested via flag — will regenerate shared keys.")
        elif not shared_keys_present:
            print("Shared proving keys missing — will generate.")
        else:
            print(
                "Keygen inputs changed since last run (Cargo.lock or "
                "--openvm-config-file) — regenerating shared keys to avoid mismatch."
            )
    else:
        print(f"Shared proving keys found at {artifacts} (up to date)")

    if args.openvm_config_file:
        print(f"OpenVM config:        {args.openvm_config_file}")
    else:
        print("OpenVM config:        standard (built-in)")

    # vmexes are cheap; always regenerate. Avoids "did I pick up my new ELF?".
    print(f"Will (re)build vmexes for {len(args.programs)} program(s).")
    print()

    did_any = False

    # Generation binaries read EDGE_OPENVM_CONFIG to override the built-in
    # standard VM config with a custom openvm.toml. Empty/unset → standard.
    gen_env = dict(os.environ)
    if args.openvm_config_file:
        gen_env["EDGE_OPENVM_CONFIG"] = args.openvm_config_file
    else:
        gen_env.pop("EDGE_OPENVM_CONFIG", None)

    # Drop `cuda` for host-side build (GPU backend not needed).
    keygen_features = ",".join(
        f for f in features.split(",") if f and f != "cuda"
    )

    # Always build convert_fixtures (needed for elf-to-vmexe). Build keygen
    # bin only if we'll run it.
    bins_to_build = ["convert_fixtures"]
    if need_shared_keygen:
        bins_to_build.append("keygen")

    for bin_name in bins_to_build:
        cargo_cmd: list[str] = ["cargo"]
        if toolchain:
            cargo_cmd.append(toolchain)
        cargo_cmd.extend(["build", "--release", "--bin", bin_name])
        if keygen_features:
            cargo_cmd.extend(["--features", keygen_features])
        print(f"  Building {bin_name}...")
        result = subprocess.run(
            cargo_cmd,
            cwd=build_workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        tail = (result.stdout + result.stderr).splitlines()[-5:]
        for line in tail:
            print(line)
        if result.returncode != 0:
            sys.exit(result.returncode)

    artifacts.mkdir(parents=True, exist_ok=True)

    if need_shared_keygen:
        keygen_cmd = [str(keygen_bin), "--output-dir", str(artifacts)]
        if args.with_deferral:
            # `--with-deferral` is gated to `evm-prove` builds inside the
            # keygen bin (the deferral tail flow lives in the same codepath
            # as root → halo2). The CLI validation above already refuses
            # --with-deferral when the worker --features exclude evm-prove.
            keygen_cmd.append("--with-deferral")
        print(
            f"  Running shared keygen → {artifacts}"
            + (" (with deferral)" if args.with_deferral else "")
        )
        proc = subprocess.Popen(
            keygen_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=gen_env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print("    " + line.rstrip())
        rc = proc.wait()
        if rc != 0:
            sys.exit(rc)
        # Provenance: copy the resolved config next to the keys so it's clear
        # what config this app_pk was built with (and reusable by generate-vm-vk).
        if args.openvm_config_file:
            (artifacts / "openvm-config.toml").write_text(
                Path(args.openvm_config_file).read_text()
            )
        else:
            (artifacts / "openvm-config.toml").unlink(missing_ok=True)
        version_file.write_text(inputs_hash)
        did_any = True

    for program in args.programs:
        name = program["name"]
        version = program["version"]
        elf_path = Path(program["path"])
        vmexe_dir = artifacts / "programs" / name / str(version)
        vmexe_dir.mkdir(parents=True, exist_ok=True)
        vmexe_path = vmexe_dir / "program.vmexe"
        print(f"  Building vmexe for ({name}, v{version}) → {vmexe_path}")
        convert_cmd = [
            str(convert_bin),
            "elf-to-vmexe",
            "--elf",
            str(elf_path),
            "--output",
            str(vmexe_path),
        ]
        # Deferral guests (e.g. verify-stark) use custom opcodes owned by the
        # deferral extension, which the standard transpiler can't parse. Point
        # the converter at the just-generated deferral cached_pk so it transpiles
        # with the deferral-enabled VM config. Reuses keygen output (no re-keygen).
        if args.with_deferral:
            convert_cmd += ["--deferral-cached-pk", str(artifacts / "deferral" / "cached_pk")]
        proc = subprocess.Popen(
            convert_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=gen_env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print("    " + line.rstrip())
        rc = proc.wait()
        if rc != 0:
            sys.exit(rc)
        did_any = True

    print("  Artifacts ready.")
    print()
    return did_any


# --- Docker build + up ---


def compose_file_args(args: Args, override_file: str) -> list[str]:
    files: list[str] = []
    if not args.worker_only:
        files.extend(["-f", "docker/docker-compose.yml"])
    files.extend(["-f", override_file])
    for extra in args.extra_compose_files:
        files.extend(["-f", extra])
    return files


def compose_command(args: Args, override_file: str) -> list[str]:
    return [
        "docker",
        "compose",
        *compose_file_args(args, override_file),
        "--profile",
        "gpu",
    ]


def build_axiom_edge_image(
    eff: dict[str, str],
    vcs_refs: dict[str, str],
    build_workspace: Path,
    docker_env: dict[str, str],
) -> None:
    print(
        f"Building axiom-edge-gpu image (FEATURES={eff['features']}, "
        f"CUDA_ARCH={eff['cuda_arch']})..."
    )
    build_args = [
        ("CARGO_BUILD_JOBS", eff["cargo_build_jobs"]),
        ("PROFILE", eff["profile"]),
        ("CUDA_ARCH", eff["cuda_arch"]),
        ("TARGET_CPU", eff["target_cpu"]),
        ("BUILDER_IMAGE", eff["builder_image"]),
        ("RUNTIME_IMAGE", eff["runtime_image"]),
        ("FEATURES", eff["features"]),
        ("JEMALLOC_SYS_WITH_MALLOC_CONF", eff["jemalloc"]),
        ("VCS_REF", vcs_refs["vcs_ref"]),
        ("OPENVM_REV", vcs_refs["openvm_rev"]),
        ("STARK_BACKEND_REV", vcs_refs["stark_rev"]),
    ]
    if eff["toolchain"]:
        build_args.append(("TOOLCHAIN_VERSION", eff["toolchain"]))

    cmd = ["docker", "build", "-f", str(DOCKER_DIR / "Dockerfile")]
    for k, v in build_args:
        cmd.extend(["--build-arg", f"{k}={v}"])
    cmd.extend(["-t", "axiom-edge-gpu", str(build_workspace)])
    subprocess.run(cmd, env=docker_env, check=True)


def docker_compose_up(
    args: Args,
    override_file: str,
    need_recreate: bool,
    docker_env: dict[str, str],
) -> None:
    cmd = compose_command(args, override_file)
    cmd.extend(["up", "-d", "--remove-orphans"])
    if need_recreate:
        cmd.append("--force-recreate")
    subprocess.run(cmd, env=docker_env, check=True)


def print_post_instructions(args: Args, override_file: str) -> None:
    print()
    print("All services started. Check status with:")
    status_cmd = compose_command(args, override_file) + ["ps"]
    print(f"  {' '.join(status_cmd)}")
    if not args.worker_only:
        print("  curl http://localhost:3000/healthz")
        print("  curl http://localhost:3000/workers")


# --- Main ---


def main() -> int:
    defaults = load_defaults()
    args = parse_args(defaults)
    validate(args)
    if not args.dry_run:
        validate_program_paths(args.programs)

    if not args.worker_host:
        sys.stderr.write("Auto-detecting private IP for worker registration...\n")
        detected = auto_detect_worker_host()
        if detected:
            sys.stderr.write(f"Detected private IP: {detected}\n")
            args.worker_host = detected
        else:
            sys.stderr.write("ERROR: Could not auto-detect IP. Use --worker-host=<IP>\n")
            return 1

    cpuset_ranges = compute_cpuset_ranges(args)
    eff = resolve_effective(args, defaults)

    build_workspace = REPO_ROOT
    vcs_refs = compute_vcs_refs(SCRIPT_DIR, build_workspace)

    config_dir = REPO_ROOT / "config" / "generated"
    override_file = "docker/docker-compose.provers.yml"
    override_path = REPO_ROOT / override_file

    env = build_env()
    manager_toml = render_manager_toml(env, args)
    worker_tomls = [
        render_service_toml(env, args, i, eff["manager_url"])
        for i in range(args.num_gpus)
    ]
    compose_override = render_compose_override(env, args, cpuset_ranges, eff["jemalloc"])

    if args.dry_run:
        rel_config = config_dir.relative_to(REPO_ROOT)
        if not args.worker_only:
            print(f"=== {rel_config}/manager.toml ===")
            print(manager_toml, end="")
            print()
        for i, content in enumerate(worker_tomls):
            print(f"=== {rel_config}/worker-{i}.toml ===")
            print(content, end="")
            print()
        print(f"=== {override_file} ===")
        print(compose_override, end="")
        print()
        print("=== docker compose command ===")
        print(" ".join(compose_command(args, override_file) + ["up", "-d", "--remove-orphans"]))
        return 0

    print_banner(args, eff, vcs_refs, cpuset_ranges)

    # Write generated files.
    config_dir.mkdir(parents=True, exist_ok=True)
    if args.persist_final_proofs_dir:
        Path(args.persist_final_proofs_dir).mkdir(parents=True, exist_ok=True)
    if args.persist_leaf_failure_app_proofs_dir:
        Path(args.persist_leaf_failure_app_proofs_dir).mkdir(parents=True, exist_ok=True)

    if not args.worker_only:
        (config_dir / "manager.toml").write_text(manager_toml)
        print(f"Generated {config_dir / 'manager.toml'}")
    for i, content in enumerate(worker_tomls):
        path = config_dir / f"worker-{i}.toml"
        path.write_text(content)
        prover_id = args.id_offset + i
        if args.worker_host:
            worker_url = f"http://{args.worker_host}:{args.worker_port_base + i}"
        else:
            worker_url = f"http://edge-worker-{i}:8001"
        print(f"Generated {path} (prover_id={prover_id}, worker_url={worker_url})")
    override_path.write_text(compose_override)
    print()
    print(f"Generated {override_file} with {args.num_gpus} workers.")
    print()

    # Environment seeded into docker build + compose (matches bash exports).
    docker_env = dict(os.environ)
    docker_env["DOCKER_BUILDKIT"] = "1"
    docker_env["COMPOSE_DOCKER_CLI_BUILD"] = "1"
    docker_env["FEATURES"] = eff["features"]
    docker_env["CUDA_ARCH"] = eff["cuda_arch"]
    # Inject the resolved artifacts dir so the compose volume mounts'
    # `${ARTIFACTS_PATH}` interpolation resolves to the same path the
    # host-side keygen wrote to. Sourced from [paths].artifacts_path in
    # defaults.toml — not read from the ambient environment.
    docker_env["ARTIFACTS_PATH"] = args.artifacts_path
    # HALO2_PK_PATH only exported when --halo2-pk-path is set. Compose
    # template gates the worker mount on the same value being non-empty.
    if args.halo2_pk_path:
        docker_env["HALO2_PK_PATH"] = args.halo2_pk_path
    # Same idea for the metrics mount (`${METRICS_OUTPUT_PATH}`) and the
    # worker log level (`${RUST_LOG}`): set them from defaults.toml so they
    # don't silently fall through to whatever happens to be in the shell.
    docker_env["METRICS_OUTPUT_PATH"] = args.metrics_output_path
    docker_env["RUST_LOG"] = args.log_level
    docker_env["VCS_REF"] = vcs_refs["vcs_ref"]
    docker_env["OPENVM_REV"] = vcs_refs["openvm_rev"]
    docker_env["STARK_BACKEND_REV"] = vcs_refs["stark_rev"]
    # Inject the program loadout into every container. Same value reaches
    # manager + workers, which both parse it on startup.
    docker_env["EDGE_PROGRAMS"] = programs_env_value(args.programs)
    if eff["toolchain"]:
        docker_env["TOOLCHAIN_VERSION"] = eff["toolchain"]
    else:
        docker_env.pop("TOOLCHAIN_VERSION", None)

    # Keygen (builds convert_fixtures host-side and runs it).
    did_keygen = ensure_artifacts(args, eff["features"], eff["toolchain"], build_workspace)
    if did_keygen:
        print(
            "Artifacts changed on disk; forcing container recreation so workers "
            "reload program.vmexe and keys."
        )

    # Build + up from the repository root. Compose files live under docker/,
    # and their relative volume paths are resolved from that directory.
    os.chdir(REPO_ROOT)
    if args.worker_only:
        print(f"Worker-only mode: starting {args.num_gpus} GPU workers (no manager/metrics)")
    else:
        print(
            f"Primary mode: starting manager + {args.num_gpus} GPU workers + metrics stack"
        )
    if not args.no_build:
        build_axiom_edge_image(eff, vcs_refs, build_workspace, docker_env)
    docker_compose_up(args, override_file, did_keygen, docker_env)

    print_post_instructions(args, override_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
