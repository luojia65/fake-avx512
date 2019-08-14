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
#[allow(non_camel_case_types)]
pub struct __m512d(f64, f64, f64, f64, f64, f64, f64, f64);
#[allow(non_camel_case_types)]
pub struct __m512(
    f32, f32, f32, f32, f32, f32, f32, f32, 
    f32, f32, f32, f32, f32, f32, f32, f32
);

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

macro_rules! impl_mask_arith_binary {
    ($($fn_name_mask: ident, $fn_name_maskz: ident, 
    $vec_type: ty, $mask_type: ty, $binary_func: ident, $zero: expr,
    [$elem: ty; $iter_cnt: expr];)*) => {
        $(
pub unsafe fn $fn_name_mask(src: $vec_type, k: $mask_type, a: $vec_type, b: $vec_type) -> $vec_type {
    let src: [$elem; $iter_cnt] = transmute(src);
    let a: [$elem; $iter_cnt] = transmute(a);
    let b: [$elem; $iter_cnt] = transmute(b);
    let mut dst: [$elem; $iter_cnt] = MaybeUninit::uninit().assume_init();
    for j in 0..$iter_cnt {
        dst[j] = if k & (0b1 << j) != 0 {
            <$elem>::$binary_func(a[j], b[j])
        } else {
            src[j]
        };
    }
    transmute(dst)
}

pub unsafe fn $fn_name_maskz(k: $mask_type, a: $vec_type, b: $vec_type) -> $vec_type {
    let a: [$elem; $iter_cnt] = transmute(a);
    let b: [$elem; $iter_cnt] = transmute(b);
    let mut dst: [$elem; $iter_cnt] = MaybeUninit::uninit().assume_init();
    for j in 0..$iter_cnt {
        dst[j] = if k & (0b1 << j) != 0 { <$elem>::$binary_func(a[j], b[j]) } else { $zero };
    }
    transmute(dst)
}
        )*
    };
}

use std::ops::{Add, Sub};

impl_mask_arith_binary! {
    _mm_mask_add_epi8, _mm_maskz_add_epi8, __m128i, __mmask8, add, 0, [i8; 16];
    _mm_mask_add_epi16, _mm_maskz_add_epi16, __m128i, __mmask8, add, 0, [i16; 8];
    _mm_mask_add_epi32, _mm_maskz_add_epi32, __m128i, __mmask8, add, 0, [i32; 4];
    _mm_mask_add_epi64, _mm_maskz_add_epi64, __m128i, __mmask8, add, 0, [i64; 2];
    _mm256_mask_add_epi8,  _mm256_maskz_add_epi8,  __m256i, __mmask32, add, 0, [i8; 32];
    _mm256_mask_add_epi16, _mm256_maskz_add_epi16, __m256i, __mmask16, add, 0, [i16; 16];
    _mm256_mask_add_epi32, _mm256_maskz_add_epi32, __m256i, __mmask8, add, 0, [i32; 8];
    _mm256_mask_add_epi64, _mm256_maskz_add_epi64, __m256i, __mmask8, add, 0, [i64; 4];
    _mm512_mask_add_epi8,  _mm512_maskz_add_epi8,  __m512i, __mmask64, add, 0, [i8; 64];
    _mm512_mask_add_epi16, _mm512_maskz_add_epi16, __m512i, __mmask32, add, 0, [i16; 32];
    _mm512_mask_add_epi32, _mm512_maskz_add_epi32, __m512i, __mmask16, add, 0, [i32; 16];
    _mm512_mask_add_epi64, _mm512_maskz_add_epi64, __m512i, __mmask8, add, 0, [i64; 8];

    _mm_mask_add_pd, _mm_maskz_add_pd, __m128d, __mmask8, add, 0.0, [f64; 2];
    _mm256_mask_add_pd, _mm256_maskz_add_pd, __m256d, __mmask8, add, 0.0, [f64; 4];
    _mm512_mask_add_pd, _mm512_maskz_add_pd, __m512d, __mmask8, add, 0.0, [f64; 8];
    _mm_mask_add_ps, _mm_maskz_add_ps, __m128, __mmask8, add, 0.0, [f32; 4];
    _mm256_mask_add_ps, _mm256_maskz_add_ps, __m256, __mmask8, add, 0.0, [f32; 8];
    _mm512_mask_add_ps, _mm512_maskz_add_ps, __m512, __mmask16, add, 0.0, [f32; 16];

    _mm_mask_adds_epi8, _mm_maskz_adds_epi8, __m128i, __mmask16, saturating_add, 0, [i8; 16];
    _mm_mask_adds_epi16, _mm_maskz_adds_epi16, __m128i, __mmask8, saturating_add, 0, [i16; 8];
    _mm_mask_adds_epu8, _mm_maskz_adds_epu8, __m128i, __mmask16, saturating_add, 0, [u8; 16];
    _mm_mask_adds_epu16, _mm_maskz_adds_epu16, __m128i, __mmask8, saturating_add, 0, [u16; 8];
    _mm256_mask_adds_epi8, _mm256_maskz_adds_epi8, __m256i, __mmask32, saturating_add, 0, [i8; 32];
    _mm256_mask_adds_epi16, _mm256_maskz_adds_epi16, __m256i, __mmask16, saturating_add, 0, [i16; 16];
    _mm256_mask_adds_epu8, _mm256_maskz_adds_epu8, __m256i, __mmask32, saturating_add, 0, [u8; 32];
    _mm256_mask_adds_epu16, _mm256_maskz_adds_epu16, __m256i, __mmask16, saturating_add, 0, [u16; 16];
    _mm512_mask_adds_epi8, _mm512_maskz_adds_epi8, __m512i, __mmask64, saturating_add, 0, [i8; 64];
    _mm512_mask_adds_epi16, _mm512_maskz_adds_epi16, __m512i, __mmask32, saturating_add, 0, [i16; 32];
    _mm512_mask_adds_epu8, _mm512_maskz_adds_epu8, __m512i, __mmask64, saturating_add, 0, [u8; 64];
    _mm512_mask_adds_epu16, _mm512_maskz_adds_epu16, __m512i, __mmask32, saturating_add, 0, [u16; 32];
    
    _mm_mask_sub_epi8, _mm_maskz_sub_epi8, __m128i, __mmask8, sub, 0, [i8; 16];
    _mm_mask_sub_epi16, _mm_maskz_sub_epi16, __m128i, __mmask8, sub, 0, [i16; 8];
    _mm_mask_sub_epi32, _mm_maskz_sub_epi32, __m128i, __mmask8, sub, 0, [i32; 4];
    _mm_mask_sub_epi64, _mm_maskz_sub_epi64, __m128i, __mmask8, sub, 0, [i64; 2];
    _mm256_mask_sub_epi8,  _mm256_maskz_sub_epi8,  __m256i, __mmask32, sub, 0, [i8; 32];
    _mm256_mask_sub_epi16, _mm256_maskz_sub_epi16, __m256i, __mmask16, sub, 0, [i16; 16];
    _mm256_mask_sub_epi32, _mm256_maskz_sub_epi32, __m256i, __mmask8, sub, 0, [i32; 8];
    _mm256_mask_sub_epi64, _mm256_maskz_sub_epi64, __m256i, __mmask8, sub, 0, [i64; 4];
    _mm512_mask_sub_epi8,  _mm512_maskz_sub_epi8,  __m512i, __mmask64, sub, 0, [i8; 64];
    _mm512_mask_sub_epi16, _mm512_maskz_sub_epi16, __m512i, __mmask32, sub, 0, [i16; 32];
    _mm512_mask_sub_epi32, _mm512_maskz_sub_epi32, __m512i, __mmask16, sub, 0, [i32; 16];
    _mm512_mask_sub_epi64, _mm512_maskz_sub_epi64, __m512i, __mmask8, sub, 0, [i64; 8];

    _mm_mask_sub_pd, _mm_maskz_sub_pd, __m128d, __mmask8, sub, 0.0, [f64; 2];
    _mm256_mask_sub_pd, _mm256_maskz_sub_pd, __m256d, __mmask8, sub, 0.0, [f64; 4];
    _mm512_mask_sub_pd, _mm512_maskz_sub_pd, __m512d, __mmask8, sub, 0.0, [f64; 8];
    _mm_mask_sub_ps, _mm_maskz_sub_ps, __m128, __mmask8, sub, 0.0, [f32; 4];
    _mm256_mask_sub_ps, _mm256_maskz_sub_ps, __m256, __mmask8, sub, 0.0, [f32; 8];
    _mm512_mask_sub_ps, _mm512_maskz_sub_ps, __m512, __mmask16, sub, 0.0, [f32; 16];

    _mm_mask_subs_epi8, _mm_maskz_subs_epi8, __m128i, __mmask16, saturating_sub, 0, [i8; 16];
    _mm_mask_subs_epi16, _mm_maskz_subs_epi16, __m128i, __mmask8, saturating_sub, 0, [i16; 8];
    _mm_mask_subs_epu8, _mm_maskz_subs_epu8, __m128i, __mmask16, saturating_sub, 0, [u8; 16];
    _mm_mask_subs_epu16, _mm_maskz_subs_epu16, __m128i, __mmask8, saturating_sub, 0, [u16; 8];
    _mm256_mask_subs_epi8, _mm256_maskz_subs_epi8, __m256i, __mmask32, saturating_sub, 0, [i8; 32];
    _mm256_mask_subs_epi16, _mm256_maskz_subs_epi16, __m256i, __mmask16, saturating_sub, 0, [i16; 16];
    _mm256_mask_subs_epu8, _mm256_maskz_subs_epu8, __m256i, __mmask32, saturating_sub, 0, [u8; 32];
    _mm256_mask_subs_epu16, _mm256_maskz_subs_epu16, __m256i, __mmask16, saturating_sub, 0, [u16; 16];
    _mm512_mask_subs_epi8, _mm512_maskz_subs_epi8, __m512i, __mmask64, saturating_sub, 0, [i8; 64];
    _mm512_mask_subs_epi16, _mm512_maskz_subs_epi16, __m512i, __mmask32, saturating_sub, 0, [i16; 32];
    _mm512_mask_subs_epu8, _mm512_maskz_subs_epu8, __m512i, __mmask64, saturating_sub, 0, [u8; 64];
    _mm512_mask_subs_epu16, _mm512_maskz_subs_epu16, __m512i, __mmask32, saturating_sub, 0, [u16; 32];
}


/*
    _mm_mask_add_round_pd, _mm_maskz_add_round_pd, __m128d, __mmask8, add, 0.0, [f64; 2];
    _mm512_mask_add_round_pd, _mm512_maskz_add_round_pd, __m512d, __mmask8, add, 0.0, [f64; 8];
    _mm_mask_add_round_ps, _mm_maskz_add_round_ps, __m128, __mmask8, add, 0.0, [f32; 4];
    _mm512_mask_add_round_ps, _mm512_maskz_add_round_ps, __m512, __mmask16, add, 0.0, [f32; 16];
*/
