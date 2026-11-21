# Changelog

All notable changes to openset-llm. Dates are YYYY-MM-DD.

## Unreleased

- First public release.
- `./serve --kv6` and `--kv5` (experimental): 6- and 5-bit attention caches. On an 8 GB machine speculation stays on
  to 12k / 16k context and plain decoding reaches 32k / 64k. Not exact; needs AVX-512; see the README's memory section.
- `./serve` now refuses two cache flags together and contexts that cannot fit with `--kv6` / `--kv5` under 12 GB.
