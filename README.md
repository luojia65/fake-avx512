# fake-avx512

[![Crates.io][crates-badge]][crates-url]
[![996ICU+WTFPL licensed][pl-badge]][pl-url]

[crates-badge]: https://img.shields.io/crates/v/mc-varint.svg
[crates-url]: https://crates.io/crates/fake-avx512
[pl-badge]: https://img.shields.io/badge/license-996ICU+WTFPL-blue.svg
[pl-url]: LICENSE

Simulate AVX-512 instructions for test or research purposes.
This crate is not performance oriented; downstream crates could use it as
a fallback implementation, other than a speedy SIMD replacement.
