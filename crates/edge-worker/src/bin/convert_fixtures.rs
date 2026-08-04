//! Utility to convert test fixtures to the format required by edge-worker.
//!
//! This binary converts:
//! 1. RISC-V ELF files to OpenVM VmExe format (program.vmexe)
//! 2. JSON input files to bincode-serialized StdIn format (input.bin)
//!
//! # Usage
//!
//! ```sh
//! # Convert ELF to VmExe
//! cargo run --release --bin convert_fixtures -- elf-to-vmexe \
//!     --elf /path/to/program.elf \
//!     --output /path/to/program.vmexe
//!
//! # Convert JSON input to bincode StdIn
//! cargo run --release --bin convert_fixtures -- json-to-stdin \
//!     --json /path/to/input.json \
//!     --output /path/to/input.bin
//!
//! # Convert all fixtures at once
//! cargo run --release --bin convert_fixtures -- all \
//!     --fixtures-dir /path/to/test-fixtures \
//!     --output-dir /path/to/output
//! ```
//!
//! Note: This binary requires the SDK dependencies and cannot be built with mock-provers.

#[cfg(not(feature = "mock-provers"))]
mod converter {
    use clap::{Parser, Subcommand};
    use edge_worker::openvm_config::create_edge_sdk;
    use eyre::{Context, Result};
    use openvm_stark_sdk::config::{
        internal_params_with_100_bits_security, leaf_params_with_100_bits_security,
    };
    use sdk_v2::config::AggregationSystemParams;
    use sdk_v2::fs::write_object_to_file;
    use sdk_v2::openvm_circuit::arch::instructions::exe::VmExe;
    use sdk_v2::types::ExecutableFormat;
    use sdk_v2::{Sdk, StdIn, F};
    use std::path::{Path, PathBuf};

    #[derive(Parser)]
    #[command(name = "convert_fixtures")]
    #[command(about = "Convert test fixtures to edge-worker format")]
    pub struct Cli {
        #[command(subcommand)]
        pub command: Commands,
    }

    #[derive(Subcommand)]
    pub enum Commands {
        /// Convert RISC-V ELF to OpenVM VmExe format
        ElfToVmexe {
            /// Path to the input ELF file
            #[arg(long)]
            elf: PathBuf,

            /// Path for the output VmExe file
            #[arg(long)]
            output: PathBuf,

            /// Optional path to a deferral `SdkCachedProvingKey` (produced by
            /// `keygen --with-deferral`). When set, the ELF is transpiled with
            /// the deferral-enabled VM config. Required for verify-stark /
            /// deferral guests: their custom opcodes are owned by the deferral
            /// extension, which the plain edge config's transpiler lacks
            /// ("couldn't parse the next instruction"). Needs `evm-prove`.
            #[arg(long)]
            deferral_cached_pk: Option<PathBuf>,
        },

        /// Convert JSON input to bincode StdIn format
        JsonToStdin {
            /// Path to the input JSON file
            #[arg(long)]
            json: PathBuf,

            /// Path for the output bincode file
            #[arg(long)]
            output: PathBuf,
        },

        /// Convert JSON input to the raw "compact" guest bytes (single element).
        ///
        /// Unlike `json-to-stdin`, this writes the version-stripped element
        /// bytes verbatim — no `StdIn` wrapper, no byte→field expansion. The
        /// result is ~4x smaller and is uploaded to workers via
        /// `/upload_input_compact`, which reconstructs the `StdIn` locally.
        /// Only single-element inputs are supported (the compact endpoint does
        /// a single `write_bytes`); multi-element inputs must use json-to-stdin.
        JsonToCompact {
            /// Path to the input JSON file
            #[arg(long)]
            json: PathBuf,

            /// Path for the output raw bytes file
            #[arg(long)]
            output: PathBuf,
        },

        /// Convert all fixtures in a directory
        All {
            /// Path to the fixtures directory
            #[arg(long)]
            fixtures_dir: PathBuf,

            /// Path for the output directory
            #[arg(long)]
            output_dir: PathBuf,
        },

        /// Generate proving keys (app_pk, agg_stark_pk) from an ELF file
        Keygen {
            /// Path to the input ELF file
            #[arg(long)]
            elf: PathBuf,

            /// Path for the output directory (will contain app_pk, agg_stark_pk, program.vmexe)
            #[arg(long)]
            output_dir: PathBuf,
        },

        /// Test the full SDK prove pipeline end-to-end (single process, no HTTP)
        TestPipeline {
            /// Path to the input ELF file
            #[arg(long)]
            elf: PathBuf,

            /// Path to the bincode-serialized StdIn input file
            #[arg(long)]
            input: PathBuf,
        },

        /// Test the prove pipeline using pre-generated keys loaded from disk
        TestWithKeys {
            /// Path to the input ELF file
            #[arg(long)]
            elf: PathBuf,

            /// Path to the bincode-serialized StdIn input file
            #[arg(long)]
            input: PathBuf,

            /// Path to the artifacts directory (containing app_pk, agg_stark_pk, program.vmexe)
            #[arg(long)]
            artifacts_dir: PathBuf,
        },
    }

    /// Convert ELF file to VmExe format.
    pub fn convert_elf_to_vmexe(
        elf_path: &PathBuf,
        output_path: &PathBuf,
        deferral_cached_pk: Option<&Path>,
    ) -> Result<()> {
        println!("Converting ELF to VmExe...");
        println!("  Input: {}", elf_path.display());
        println!("  Output: {}", output_path.display());

        // Read ELF file
        let elf_bytes = std::fs::read(elf_path)
            .wrap_err_with(|| format!("Failed to read ELF file: {}", elf_path.display()))?;

        println!("  ELF file read: {} bytes", elf_bytes.len());

        // Convert ELF bytes to VmExe using the SDK's convert_to_exe.
        // The ExecutableFormat::from(&[u8]) handles ELF decoding.
        let executable: ExecutableFormat = elf_bytes.as_slice().into();
        let exe = if let Some(pk_path) = deferral_cached_pk {
            #[cfg(feature = "evm-prove")]
            {
                use openvm_sdk_config::SdkVmConfig;
                use sdk_v2::fs::read_object_from_file;
                use sdk_v2::keygen::SdkCachedProvingKey;
                println!(
                    "  Using deferral-enabled transpiler (cached_pk: {})",
                    pk_path.display()
                );
                let cached_pk: SdkCachedProvingKey<SdkVmConfig> = read_object_from_file(pk_path)
                    .wrap_err_with(|| {
                        format!("Failed to read deferral cached_pk: {}", pk_path.display())
                    })?;
                let sdk = Sdk::from_deferral_cached_proving_key(cached_pk).map_err(|e| {
                    eyre::eyre!("Failed to reconstruct deferral SDK from cached_pk: {e}")
                })?;
                sdk.convert_to_exe(executable)
                    .wrap_err("Failed to convert ELF to VmExe (deferral config)")?
            }
            #[cfg(not(feature = "evm-prove"))]
            {
                // The deferral keyset is STARK-level; the vmexe only depends on
                // the deferral VM *config* (the transpiler), not the proving key
                // (`convert_to_exe` never reads the pk). Rebuild that config with
                // the same non-evm constructor `keygen --with-deferral` used, so a
                // stark-only (`--halo2 none`) deferral deployment can transpile
                // without pulling in `evm-prove`.
                use edge_worker::openvm_config::create_edge_sdk_with_deferral;
                let _ = pk_path;
                println!("  Using deferral-enabled transpiler (stark-only; VM config only)");
                let sdk = create_edge_sdk_with_deferral()?;
                sdk.convert_to_exe(executable)
                    .wrap_err("Failed to convert ELF to VmExe (deferral config)")?
            }
        } else {
            // Standard axiom-edge VM settings (reth benchmark degree/public values).
            let sdk = create_edge_sdk()?;
            sdk.convert_to_exe(executable)
                .wrap_err("Failed to convert ELF to VmExe")?
        };

        println!("  VmExe created successfully");
        println!("    Program size: {} instructions", exe.program.len());

        // Write VmExe to file (dereference Arc to get the inner VmExe)
        write_object_to_file(output_path, exe.as_ref())
            .wrap_err_with(|| format!("Failed to write VmExe to: {}", output_path.display()))?;

        println!("  Wrote VmExe to: {}", output_path.display());
        Ok(())
    }

    /// Convert JSON input to bincode StdIn format.
    ///
    /// Expected JSON format (from openvm-reth-benchmark):
    /// ```json
    /// {
    ///   "input": ["0x01<hex_encoded_bytes>", ...]
    /// }
    /// ```
    ///
    /// The hex string starts with "0x01" where "01" is a version/format byte,
    /// followed by the OpenVM serde-serialized data as hex-encoded bytes.
    pub fn convert_json_to_stdin(json_path: &PathBuf, output_path: &PathBuf) -> Result<()> {
        println!("Converting JSON to bincode StdIn...");
        println!("  Input: {}", json_path.display());
        println!("  Output: {}", output_path.display());

        // Read JSON file
        let json_str = std::fs::read_to_string(json_path)
            .wrap_err_with(|| format!("Failed to read JSON file: {}", json_path.display()))?;

        // Parse JSON
        let json: serde_json::Value =
            serde_json::from_str(&json_str).wrap_err("Failed to parse JSON")?;

        // Extract input array
        let input_array = json
            .get("input")
            .ok_or_else(|| eyre::eyre!("JSON missing 'input' field"))?
            .as_array()
            .ok_or_else(|| eyre::eyre!("'input' field is not an array"))?;

        println!("  Found {} input elements", input_array.len());

        // Create StdIn and add each input
        let mut stdin = StdIn::default();

        for (i, item) in input_array.iter().enumerate() {
            let hex_str = item
                .as_str()
                .ok_or_else(|| eyre::eyre!("Input element {} is not a string", i))?;

            // Remove 0x prefix if present
            let hex_str = hex_str.strip_prefix("0x").unwrap_or(hex_str);

            // Decode hex to bytes
            let bytes = hex::decode(hex_str)
                .wrap_err_with(|| format!("Failed to decode hex for element {}", i))?;

            // The JSON format from openvm-reth-benchmark includes a version byte (0x01) prefix.
            // Skip it to get the actual OpenVM serde-serialized data.
            let data_bytes = if bytes.first() == Some(&0x01) && bytes.len() > 1 {
                println!("  Element {}: stripping version byte prefix (0x01)", i);
                &bytes[1..]
            } else {
                &bytes[..]
            };

            // Write bytes to stdin
            stdin.write_bytes(data_bytes);
            println!(
                "  Element {}: {} bytes written to stdin",
                i,
                data_bytes.len()
            );
        }

        // Serialize to bincode
        let bincode_bytes =
            bincode::serialize(&stdin).wrap_err("Failed to serialize StdIn to bincode")?;

        println!("  StdIn serialized: {} bytes", bincode_bytes.len());

        // Write to output file
        std::fs::write(output_path, &bincode_bytes)
            .wrap_err_with(|| format!("Failed to write to: {}", output_path.display()))?;

        println!("  Wrote bincode StdIn to: {}", output_path.display());
        Ok(())
    }

    /// Convert JSON input to raw "compact" guest bytes (single element only).
    ///
    /// Mirrors the per-element extraction in `convert_json_to_stdin` (hex
    /// decode + version-byte strip) but writes the bytes verbatim instead of
    /// wrapping them in a `StdIn`. The worker's `/upload_input_compact` endpoint
    /// performs the equivalent `stdin.write_bytes` reconstruction, so for a
    /// single-element input the resulting on-worker `StdIn` is identical.
    pub fn convert_json_to_compact(json_path: &PathBuf, output_path: &PathBuf) -> Result<()> {
        println!("Converting JSON to compact guest bytes...");
        println!("  Input: {}", json_path.display());
        println!("  Output: {}", output_path.display());

        let json_str = std::fs::read_to_string(json_path)
            .wrap_err_with(|| format!("Failed to read JSON file: {}", json_path.display()))?;

        let json: serde_json::Value =
            serde_json::from_str(&json_str).wrap_err("Failed to parse JSON")?;

        let input_array = json
            .get("input")
            .ok_or_else(|| eyre::eyre!("JSON missing 'input' field"))?
            .as_array()
            .ok_or_else(|| eyre::eyre!("'input' field is not an array"))?;

        // The compact endpoint does a single `write_bytes`, so it can only
        // represent one element. Reject multi-element inputs explicitly rather
        // than silently merging them (which would corrupt the guest's reads).
        if input_array.len() != 1 {
            eyre::bail!(
                "compact conversion supports exactly one input element, found {} \
                 (use json-to-stdin for multi-element inputs)",
                input_array.len()
            );
        }

        let hex_str = input_array[0]
            .as_str()
            .ok_or_else(|| eyre::eyre!("Input element 0 is not a string"))?;
        let hex_str = hex_str.strip_prefix("0x").unwrap_or(hex_str);
        let bytes = hex::decode(hex_str).wrap_err("Failed to decode hex for element 0")?;

        // Same version-byte (0x01) strip as convert_json_to_stdin.
        let data_bytes = if bytes.first() == Some(&0x01) && bytes.len() > 1 {
            println!("  Element 0: stripping version byte prefix (0x01)");
            &bytes[1..]
        } else {
            &bytes[..]
        };

        std::fs::write(output_path, data_bytes)
            .wrap_err_with(|| format!("Failed to write to: {}", output_path.display()))?;

        println!(
            "  Wrote {} compact bytes to: {}",
            data_bytes.len(),
            output_path.display()
        );
        Ok(())
    }

    /// Convert all fixtures in a directory.
    pub fn convert_all(fixtures_dir: &PathBuf, output_dir: &PathBuf) -> Result<()> {
        println!("Converting all fixtures...");
        println!("  Fixtures dir: {}", fixtures_dir.display());
        println!("  Output dir: {}", output_dir.display());

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Look for ELF file
        let elf_path = fixtures_dir.join("openvm-client-eth.elf");
        if elf_path.exists() {
            let vmexe_output = output_dir.join("program.vmexe");
            convert_elf_to_vmexe(&elf_path, &vmexe_output, None)?;
        } else {
            println!("  WARNING: ELF file not found at {}", elf_path.display());
        }

        // Look for JSON input files
        for entry in std::fs::read_dir(fixtures_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                let stem = path.file_stem().unwrap().to_string_lossy();
                let output_path = output_dir.join(format!("{}.bin", stem));
                convert_json_to_stdin(&path, &output_path)?;
            }
        }

        // Also create the standard input.bin from the first JSON found
        let json_path = fixtures_dir.join("24000481.json");
        if json_path.exists() {
            let input_output = output_dir.join("input.bin");
            if !input_output.exists() {
                convert_json_to_stdin(&json_path, &input_output)?;
            }
        }

        println!("\nConversion complete!");
        println!("Output files are in: {}", output_dir.display());
        Ok(())
    }

    /// Generate proving keys from an ELF file.
    ///
    /// This creates:
    /// - app_pk: Application proving key
    /// - agg_stark_pk: Aggregation STARK proving key
    /// - program.vmexe: VM executable (transpiled from ELF)
    pub fn keygen(elf_path: &PathBuf, output_dir: &PathBuf) -> Result<()> {
        println!("Generating proving keys...");
        println!("  ELF: {}", elf_path.display());
        println!("  Output dir: {}", output_dir.display());

        std::fs::create_dir_all(output_dir)?;

        // Read ELF file
        let elf_bytes = std::fs::read(elf_path)
            .wrap_err_with(|| format!("Failed to read ELF file: {}", elf_path.display()))?;
        println!("  ELF file read: {} bytes", elf_bytes.len());

        // Build SDK with axiom-edge VM settings (matches reth benchmark degree/public values).
        let sdk = create_edge_sdk()?;

        // Convert ELF to VmExe
        println!("  Converting ELF to VmExe...");
        let executable: ExecutableFormat = elf_bytes.as_slice().into();
        let exe = sdk
            .convert_to_exe(executable)
            .wrap_err("Failed to convert ELF to VmExe")?;
        println!("    Program size: {} instructions", exe.program.len());

        // Write VmExe
        let vmexe_path = output_dir.join("program.vmexe");
        write_object_to_file(&vmexe_path, exe.as_ref())
            .wrap_err("Failed to write program.vmexe")?;
        println!("  Wrote program.vmexe");

        // Generate app proving key
        println!("  Generating app proving key (this may take a while)...");
        let (app_pk, _app_vk) = sdk.app_keygen();
        println!("  App proving key generated");

        // Write app_pk
        let app_pk_path = output_dir.join("app_pk");
        write_object_to_file(&app_pk_path, &app_pk).wrap_err("Failed to write app_pk")?;
        println!("  Wrote app_pk");

        // Generate aggregation proving key
        println!("  Generating aggregation proving key (this may take a while)...");
        let agg_pk = sdk.agg_pk();
        println!("  Aggregation proving key generated");

        // Write agg_stark_pk
        let agg_pk_path = output_dir.join("agg_stark_pk");
        write_object_to_file(&agg_pk_path, &agg_pk).wrap_err("Failed to write agg_stark_pk")?;
        println!("  Wrote agg_stark_pk");

        // Generate the verification baseline. It commits to the app exe and to
        // each aggregation layer's vk, so it identifies this program under this
        // deployment's VM config and a caller verifies a final proof against it
        // without holding the ELF. Both keys above are cached on the SDK, so
        // this reuses them rather than running keygen again.
        //
        // The exe is normalized through the same bitcode encoding `program.vmexe`
        // uses on disk, so the baseline commits to the exe the workers load
        // rather than the in-memory one (see `stark_verify::build_vm_vk_from_elf`).
        println!("  Generating verification baseline...");
        let exe_bytes =
            bitcode::serialize(exe.as_ref()).wrap_err("Failed to bitcode-serialize VmExe")?;
        let normalized_exe: VmExe<F> =
            bitcode::deserialize(&exe_bytes).wrap_err("Failed to bitcode-deserialize VmExe")?;
        let baseline = sdk
            .prover(normalized_exe)
            .wrap_err("Failed to build the stark prover for the baseline")?
            .generate_baseline();
        let baseline_path = output_dir.join("baseline.bin");
        std::fs::write(
            &baseline_path,
            bitcode::serialize(&baseline).wrap_err("Failed to encode the baseline")?,
        )
        .wrap_err("Failed to write baseline.bin")?;
        println!("  Wrote baseline.bin");

        println!("\nKeygen complete! Output files:");
        println!("  {}", vmexe_path.display());
        println!("  {}", app_pk_path.display());
        println!("  {}", agg_pk_path.display());
        println!("  {}", baseline_path.display());
        Ok(())
    }

    /// Test the full SDK prove pipeline end-to-end in a single process.
    ///
    /// This uses the SDK's StarkProver::prove which runs:
    /// app_prove → leaf → internal_for_leaf → internal_recursive → 2 wraps → compress
    /// all in memory without any serialization.
    pub fn test_pipeline(elf_path: &PathBuf, input_path: &PathBuf) -> Result<()> {
        println!("Testing full SDK prove pipeline...");
        println!("  ELF: {}", elf_path.display());
        println!("  Input: {}", input_path.display());

        // Read ELF file
        let elf_bytes = std::fs::read(elf_path)
            .wrap_err_with(|| format!("Failed to read ELF file: {}", elf_path.display()))?;
        println!("  ELF file read: {} bytes", elf_bytes.len());

        // Read input file (bincode-serialized StdIn)
        let input_bytes = std::fs::read(input_path)
            .wrap_err_with(|| format!("Failed to read input file: {}", input_path.display()))?;
        let stdin: StdIn = bincode::deserialize(&input_bytes)
            .wrap_err("Failed to deserialize StdIn from bincode")?;
        println!("  Input deserialized successfully");

        // Create SDK with the same configuration as keygen.
        let sdk = create_edge_sdk()?;

        // Convert ELF to executable
        let executable: ExecutableFormat = elf_bytes.as_slice().into();
        let exe = sdk
            .convert_to_exe(executable)
            .wrap_err("Failed to convert ELF to VmExe")?;
        println!("  VmExe created: {} instructions", exe.program.len());

        // Run the full prove pipeline using the SDK
        println!("\n  Starting SDK prove (this will take a while)...");
        println!(
            "  Pipeline: app → leaf → internal_for_leaf → internal_recursive → 2 wraps → compress"
        );
        let start = std::time::Instant::now();

        let (proof, baseline) = sdk.prove(exe, stdin, &[]).wrap_err("SDK prove failed")?;

        let elapsed = start.elapsed();
        println!("\n  SDK prove completed in {:?}", elapsed);

        // Verify the proof
        println!("\n  Verifying proof...");
        let agg_vk = sdk.agg_vk();
        Sdk::verify_proof((*agg_vk).clone(), baseline, &proof)
            .wrap_err("Proof verification failed")?;
        println!("  Proof verified successfully!");

        println!("\n  Pipeline test PASSED!");
        Ok(())
    }

    /// Test the prove pipeline with pre-generated keys loaded from disk.
    ///
    /// This loads the same artifacts that the service uses (app_pk, agg_stark_pk),
    /// creates provers the SDK way (AggProver::from_pk), and runs the full pipeline.
    /// This isolates whether the issue is in the keys/artifacts or in the distributed pipeline.
    pub fn test_with_keys(elf_path: &Path, input_path: &Path, artifacts_dir: &Path) -> Result<()> {
        use openvm_sdk_config::SdkVmConfig;
        use sdk_v2::keygen::{AggProvingKey, AppProvingKey};
        use sdk_v2::prover::{AggProver, StarkProver};
        use std::sync::Arc;

        println!("Testing prove pipeline with pre-generated keys...");
        println!("  ELF: {}", elf_path.display());
        println!("  Input: {}", input_path.display());
        println!("  Artifacts: {}", artifacts_dir.display());

        // Load app_pk
        let app_pk_path = artifacts_dir.join("app_pk");
        println!("  Loading app_pk from {}...", app_pk_path.display());
        let app_pk: AppProvingKey<SdkVmConfig> = sdk_v2::fs::read_object_from_file(&app_pk_path)?;
        println!("  app_pk loaded");

        // Load agg_stark_pk
        let agg_pk_path = artifacts_dir.join("agg_stark_pk");
        println!("  Loading agg_stark_pk from {}...", agg_pk_path.display());
        let agg_pk: AggProvingKey = sdk_v2::fs::read_object_from_file(&agg_pk_path)?;
        println!("  agg_stark_pk loaded");
        // Load VmExe
        let exe_path = artifacts_dir.join("program.vmexe");
        println!("  Loading program.vmexe from {}...", exe_path.display());
        let exe: sdk_v2::openvm_circuit::arch::instructions::exe::VmExe<F> =
            sdk_v2::fs::read_object_from_file(&exe_path)?;
        let exe = Arc::new(exe);
        println!("  program.vmexe loaded: {} instructions", exe.program.len());

        // Read input file (bincode-serialized StdIn)
        let input_bytes = std::fs::read(input_path)
            .wrap_err_with(|| format!("Failed to read input file: {}", input_path.display()))?;
        let stdin: StdIn = bincode::deserialize(&input_bytes)
            .wrap_err("Failed to deserialize StdIn from bincode")?;
        println!("  Input deserialized successfully");

        // Create provers using the SDK's from_pk constructors
        // This mirrors what the service SHOULD do if it used AggProver::from_pk
        let app_vk = Arc::new(app_pk.get_app_vk().vk);
        println!("\n  Creating AggProver from loaded keys...");
        let agg_tree_config = sdk_v2::config::AggregationTreeConfig::default();
        let agg_prover = Arc::new(AggProver::from_pk(
            app_vk.clone(),
            agg_pk.clone(),
            agg_tree_config,
            None,
        ));
        println!("  AggProver created");

        // Create the StarkProver using loaded keys.
        println!("  Creating StarkProver...");
        let agg_params = AggregationSystemParams {
            leaf: leaf_params_with_100_bits_security(),
            internal: internal_params_with_100_bits_security(),
        };
        let sdk = Sdk::builder()
            .app_pk(app_pk)
            .agg_params(agg_params)
            .build()?;

        let vm_builder = *sdk.app_vm_builder();
        let app_pk_ref = sdk.app_pk();
        let mut stark_prover = StarkProver::<sdk_v2::DefaultStarkEngine, _>::new(
            vm_builder,
            &app_pk_ref.app_vm_pk,
            exe,
            agg_prover,
            sdk_v2::DeferralSetup::Disabled,
        )?;
        println!("  StarkProver created");

        // Run the full prove pipeline
        println!("\n  Starting prove with loaded keys (this will take a while)...");
        let start = std::time::Instant::now();

        let (proof, _metadata) = stark_prover
            .prove(stdin, &[])
            .wrap_err("Prove with loaded keys failed")?;

        let elapsed = start.elapsed();
        println!("\n  Prove completed in {:?}", elapsed);

        // Verify the proof
        println!("  Verifying proof...");
        let agg_vk = sdk.agg_vk();
        let baseline = stark_prover.generate_baseline();
        Sdk::verify_proof((*agg_vk).clone(), baseline, &proof)
            .wrap_err("Proof verification failed")?;
        println!("  Proof verified successfully!");

        println!("\n  Test with loaded keys PASSED!");
        Ok(())
    }

    pub fn run() -> Result<()> {
        let cli = Cli::parse();

        match cli.command {
            Commands::ElfToVmexe {
                elf,
                output,
                deferral_cached_pk,
            } => convert_elf_to_vmexe(&elf, &output, deferral_cached_pk.as_deref()),
            Commands::JsonToStdin { json, output } => convert_json_to_stdin(&json, &output),
            Commands::JsonToCompact { json, output } => convert_json_to_compact(&json, &output),
            Commands::All {
                fixtures_dir,
                output_dir,
            } => convert_all(&fixtures_dir, &output_dir),
            Commands::Keygen { elf, output_dir } => keygen(&elf, &output_dir),
            Commands::TestPipeline { elf, input } => test_pipeline(&elf, &input),
            Commands::TestWithKeys {
                elf,
                input,
                artifacts_dir,
            } => test_with_keys(&elf, &input, &artifacts_dir),
        }
    }
}

#[cfg(not(feature = "mock-provers"))]
fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    converter::run()
}

#[cfg(feature = "mock-provers")]
fn main() {
    eprintln!("Error: convert_fixtures cannot be built with mock-provers feature");
    eprintln!("Build with: cargo build --release --bin convert_fixtures");
    std::process::exit(1);
}
