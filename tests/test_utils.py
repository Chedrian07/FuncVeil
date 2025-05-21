"""
유틸리티 함수 테스트
"""

import pytest
import numpy as np
import hashlib
import hmac
from funcveil.utils import (
    set_seed, 
    normalize_range, 
    generate_parameters_from_seed,
    create_hmac,
    truncate_hmac,
    laplace_noise,
    calculate_sensitivity
)


def test_set_seed():
    """시드 설정 테스트"""
    set_seed(42)
    value1 = np.random.random()
    
    set_seed(42)
    value2 = np.random.random()
    
    assert value1 == value2
    
    set_seed(43)
    value3 = np.random.random()
    assert value1 != value3


def test_normalize_range():
    """범위 정규화 테스트"""
    # 기본 정규화 (0-1 범위)
    assert normalize_range(5, 0, 10) == 0.5
    assert normalize_range(0, 0, 10) == 0.0
    assert normalize_range(10, 0, 10) == 1.0
    
    # 다른 범위로 정규화
    assert normalize_range(5, 0, 10, -1, 1) == 0.0
    assert normalize_range(0, 0, 10, -1, 1) == -1.0
    assert normalize_range(10, 0, 10, -1, 1) == 1.0
    
    # 동일한 최소/최대값 처리
    assert normalize_range(5, 5, 5) == 0  # 기본 타겟 최소값


def test_generate_parameters():
    """파라미터 생성 테스트"""
    params1 = generate_parameters_from_seed(42, 4)
    assert len(params1) == 4
    assert all(isinstance(p, float) for p in params1)
    
    # 동일 시드는 동일 결과 생성
    params2 = generate_parameters_from_seed(42, 4)
    assert params1 == params2
    
    # 다른 시드는 다른 결과 생성
    params3 = generate_parameters_from_seed(43, 4)
    assert params1 != params3
def test_hmac_functions():
    """HMAC 관련 함수 테스트"""
    # HMAC 생성
    hmac_value = create_hmac("test_data", "test_key")
    assert isinstance(hmac_value, str)
    assert len(hmac_value) == 64  # SHA-256 해시 길이
    
    # 동일 입력은 동일 출력 생성
    hmac_value2 = create_hmac("test_data", "test_key")
    assert hmac_value == hmac_value2
    
    # 다른 키는 다른 해시 생성
    hmac_value3 = create_hmac("test_data", "different_key")
    assert hmac_value != hmac_value3
    
    # 잘라내기
    truncated = truncate_hmac(hmac_value, 8)
    assert len(truncated) == 8
    assert truncated == hmac_value[:8]


def test_laplace_noise():
    """라플라스 노이즈 테스트"""
    # 단일 노이즈
    noise = laplace_noise(1.0)
    assert isinstance(noise, float)
    
    # 여러 노이즈
    noises = laplace_noise(1.0, 10)
    assert isinstance(noises, np.ndarray)
    assert len(noises) == 10
    
    # 스케일에 따른 분산 확인
    small_scale_noises = laplace_noise(0.1, 1000)
    large_scale_noises = laplace_noise(10.0, 1000)
    
    assert np.var(small_scale_noises) < np.var(large_scale_noises)


def test_calculate_sensitivity():
    """민감도 계산 테스트"""
    # 테스트 도함수: f'(x) = 2x
    def derivative_func(x):
        return 2 * x
    
    sensitivity = calculate_sensitivity(derivative_func, -5, 5)
    assert sensitivity == 10  # max(|2x|) for x in [-5, 5] is |2*5| = 10
