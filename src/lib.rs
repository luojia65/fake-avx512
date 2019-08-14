use core::mem::{transmute, MaybeUninit};
use std::arch::x86_64::*;

#[allow(non_camel_case_types)]
pub type __mmask8 = i8;
#[allow(non_camel_case_types)]
pub type __mmask16 = i16;
#[allow(non_camel_case_types)]
pub type __mmask32 = i32;
#[allow(non_camel_case_types)]
pub type __mmask64 = i64;

#[allow(non_camel_case_types)]
pub struct __m512i(i64, i64, i64, i64, i64, i64, i64, i64);

macro_rules! impl_mask_arith_abs {
    ($($fn_name_mask: ident, $fn_name_maskz: ident, 
    $vec_type: ty, $mask_type: ty, [$elem: ty; $iter_cnt: expr];)*) => {
        $(
pub unsafe fn $fn_name_mask(src: $vec_type, k: $mask_type, a: $vec_type) -> $vec_type {
    let src: [$elem; $iter_cnt] = transmute(src);
    let a: [$elem; $iter_cnt] = transmute(a);
    let mut dst: [$elem; $iter_cnt] = MaybeUninit::uninit().assume_init();
    for j in 0..$iter_cnt {
        dst[j] = if k & (0b1 << j) != 0 {
            a[j].abs()
        } else {
            src[j]
        };
    }
    transmute(dst)
}

pub unsafe fn $fn_name_maskz(k: $mask_type, a: $vec_type) -> $vec_type {
    let a: [$elem; $iter_cnt] = transmute(a);
    let mut dst: [$elem; $iter_cnt] = MaybeUninit::uninit().assume_init();
    for j in 0..$iter_cnt {
        dst[j] = if k & (0b1 << j) != 0 { a[j].abs() } else { 0 };
    }
    transmute(dst)
}
        )*
    };
}

impl_mask_arith_abs! {
    _mm_mask_abs_epi8,  _mm_maskz_abs_epi8,  __m128i, __mmask8, [i8; 16];
    _mm_mask_abs_epi16, _mm_maskz_abs_epi16, __m128i, __mmask8, [i16; 8];
    _mm_mask_abs_epi32, _mm_maskz_abs_epi32, __m128i, __mmask8, [i32; 4];
    _mm_mask_abs_epi64, _mm_maskz_abs_epi64, __m128i, __mmask8, [i64; 2];
    _mm256_mask_abs_epi8,  _mm256_maskz_abs_epi8,  __m256i, __mmask32, [i8; 32];
    _mm256_mask_abs_epi16, _mm256_maskz_abs_epi16, __m256i, __mmask16, [i16; 16];
    _mm256_mask_abs_epi32, _mm256_maskz_abs_epi32, __m256i, __mmask8, [i32; 8];
    _mm256_mask_abs_epi64, _mm256_maskz_abs_epi64, __m256i, __mmask8, [i64; 4];
    _mm512_mask_abs_epi8,  _mm512_maskz_abs_epi8,  __m512i, __mmask64, [i8; 64];
    _mm512_mask_abs_epi16, _mm512_maskz_abs_epi16, __m512i, __mmask32, [i16; 32];
    _mm512_mask_abs_epi32, _mm512_maskz_abs_epi32, __m512i, __mmask16, [i32; 16];
    _mm512_mask_abs_epi64, _mm512_maskz_abs_epi64, __m512i, __mmask8, [i64; 8];
}
