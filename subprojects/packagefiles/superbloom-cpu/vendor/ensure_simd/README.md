[![crates.io](https://img.shields.io/crates/v/ensure_simd.svg)](https://crates.io/crates/ensure_simd)
[![docs.rs](https://img.shields.io/docsrs/ensure_simd.svg)](https://docs.rs/ensure_simd)

# `ensure_simd`: compile-time check whether NEON or AVX2 is available

Many of my tools use AVX2 or NEON instructions performance reasons.
While NEON is automatically enabled on aarch64 architectures, AVX2 is not
enabled by default on x64,
even though your system is very likely to support it.

This library does a compile-time check that the target architecture indeed
supports either NEON or AVX2 instructions.
This is especially relevant for CI builds, where it is easy to
mis-configure the build flags and accidentally build a binary without AVX2 support.

In case the AVX2 feature is not enabled on x64, crates like `wide` and the
`portable-simd` feature will automatically fall back to scalar or 128-bit SIMD
instructions, which are less efficient than the intended 256-bit AVX2 instructions.
Thus, this crate ensures that compiled binaries actually use the intended fast-path.

If you intentionally target x86 machines without AVX2 support,
the check can be manually disabled by enabling the `scalar` feature.
Then, non-AVX2 fallbacks will be used.

The `ensure_simd` function can be used at the start of `main()` to do a
run-time check that the CPU that is running the binary actually supports
AVX2 instructions.

[This blog post](https://curiouscoding.nl/posts/distributing-rust-simd-binaries/) contains some more background.

## Installing a binary using SIMD instructions
On aarch64, NEON is always available, and you should not run into any issues --
`carg install <tool>` should _just work_.

On x64, you will need to manually instruct `cargo` to use the instruction sets available on your architecture:

``` sh
RUSTFLAGS="-C target-cpu=native" cargo install <tool>
```

Alternatively, if you prefer a more portable binary (e.g. in case the build
machine supports AVX512 but you plan to copy the binary to less fancy machines), do:

``` sh
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo install <tool>
```

If your machine is very (>10 years) old and does not support `x86-64-v3`
([wikipedia](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels)), and
thus no AVX2, you can also explicitly ask to build without AVX2 instructions,
but this will give reduced performance:

``` sh
cargo install <tool> -F scalar
```

## Distributing binaries using SIMD instructions
For maximal performance, we recommend to use `target-cpu=native` in the
repository-local configuration:

``` toml
# .cargo/config.toml
[build]
# By default, we want maximum performance rather than portability.
rustflags = ["-C", "target-cpu=native"]
```

But for CI builds that produce distributed binaries (for GitHub releases,
bioconda, pypi, ...), we instead recommend more conservative defaults:

``` toml
# .cargo/config-portable.toml
[target.'cfg(target_arch="x86_64")']
# x86-64-v2 does not have AVX2, but we need that.
# x86-64-v4 has AVX512 which we explicitly do not include for portability.
rustflags = ["-C", "target-cpu=x86-64-v3"]

[target.'cfg(all(target_arch="aarch64", target_os="macos"))']
# For aarch64 macos builds, specifically target M1 rather than generic aarch64.
rustflags = ["-C", "target-cpu=apple-a14"]
```

Then, in your workflow configuration, run `mv .cargo/config-portable.toml
.cargo/config.toml` before invoking `cargo build`.
