use fake_avx512::*;
use std::arch::x86_64::*;
use core::mem::transmute;

#[test]
fn test_mask_abs() {
    let src = [7777i64, 777, 77, 7];
    let a = [100i64, -200, 300, -400];
    let src = unsafe { _mm256_loadu_si256(&src as *const _ as *const __m256i) };
    let a = unsafe { _mm256_loadu_si256(&a as *const _ as *const __m256i) };
    let k = 0b0000_1_0_1_1;

    let dst = unsafe { _mm256_mask_abs_epi64(src, k, a) };
    let dst: [i64; 4] = unsafe { transmute(dst) };
    let expected = [100, 200, 77, 400];
    
    assert_eq!(dst, expected);
}
