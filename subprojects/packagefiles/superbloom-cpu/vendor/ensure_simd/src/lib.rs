//! This library does a compile-time check that the target architecture supports either 128-bit
//! NEON instructions (always available on aarch64), or 256-bit AVX2 instructions (must be explicitly enabled on x86-64).
//!
//! In case the AVX2 feature is not enabled on x64, crates like `wide` and the
//! `portable-simd` feature will automatically fall back to scalar or 128-bit SIMD
//! instructions, which are less efficient than the intended 256-bit AVX2 instructions.
//! Thus, this crate ensures that compiled binaries actually use the intended fast-path.
//!
//! This check is skipped in debug and documentation builds. In release builds,
//! it can be manually disabled by enabling the `scalar` feature,
//! in which case non-AVX2 fallbacks will be used.
//!
//! The [`ensure_simd`] function can be used at the start of `main()` to do a
//! run-time check that the CPU that is running the binary actually supports
//! AVX2 instructions.
//!
//! See the github readme for more details:
//! <https://github.com/ragnargrootkoerkamp/ensure_simd>.

#[cfg(not(any(
    doc,
    debug_assertions,
    target_feature = "avx2",
    target_feature = "neon",
    feature = "scalar"
)))]
compile_error!("
The tool you are trying to build uses AVX2 (on x64) or NEON (on aarch64) SIMD instructions for performance.
Unfortunately, AVX2 is not enabled by default on x64.
To get the expected performance, compile/install using e.g.:
RUSTFLAGS=\"-C target-cpu=native\" cargo ...
Alternatively, silence this error by activating the `scalar` feature (eg `cargo install -F scalar ...`).
See the readme at https://github.com/ragnargrootkoerkamp/ensure_simd for details."
);

/// Do a run-time check that AVX2 SIMD instructions are available when compiled into the binary.
///
/// Ideally call this at the very start of your `main` function, to avoid hitting illegal AVX2 instructions during e.g. argument parsing.
///
/// (NEON instructions are always available on ARM targets, so no check is needed.)
pub fn ensure_simd() {
    #[cfg(target_feature = "avx2")]
    {
        if !is_x86_feature_detected!("avx2") {
            eprintln!(
                "
This binary was compiled with AVX2 instructions enabled, but your CPU does not support this.
Please run on a CPU that supports AVX2, or build from source with the `-F scalar` feature enabled.
See the readme at https://github.com/ragnargrootkoerkamp/ensure_simd for details.
"
            );
            std::process::exit(1);
        }
    }
}
