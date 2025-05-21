"""
기본 마스킹 클래스 테스트
"""

import pytest
import numpy as np
from funcveil.masking.base import IFitMask
from funcveil.masking.sin import SinMask
from funcveil.masking.logexp import LogExpMask
from funcveil.masking.composite import CompositeMask
from funcveil.masking.hmac_mask import HMACMask


def test_sin_mask_creation():
    """SinMask 생성 테스트"""
    mask = SinMask(seed=42)
    assert mask is not None
    assert mask.initialized is False


def test_sin_mask_fit():
    """SinMask fit 테스트"""
    mask = SinMask(seed=42)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    assert mask.initialized is True
    assert mask.bounds == (10, 50)
    assert 'A' in mask.params
    assert 'omega' in mask.params
    assert 'phi' in mask.params
    assert 'B' in mask.params


def test_sin_mask_apply():
    """SinMask 적용 테스트"""
    mask = SinMask(seed=42)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    # 마스킹 적용
    masked_value = mask.mask(30)
    assert isinstance(masked_value, float)
    
    # 동일한 입력에 대한 일관성 확인
    masked_value2 = mask.mask(30)
    assert masked_value == masked_value2
    
    # 다른 입력에 대해 다른 출력 확인
    masked_value3 = mask.mask(40)
    assert masked_value != masked_value3


def test_sin_mask_derivative():
    """SinMask 도함수 테스트"""
    mask = SinMask(seed=42)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    # 도함수 값 계산
    deriv = mask.derivative(30)
    assert isinstance(deriv, float)
def test_sin_mask_sensitivity():
    """SinMask 민감도 계산 테스트"""
    mask = SinMask(seed=42)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    # 민감도 계산
    sensitivity = mask.get_sensitivity()
    assert isinstance(sensitivity, float)
    assert sensitivity > 0


def test_logexp_mask():
    """LogExpMask 테스트"""
    mask = LogExpMask(seed=42)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    assert mask.initialized is True
    
    # 로그 모드
    log_mask = LogExpMask(seed=42, use_log=True)
    log_mask.fit(data)
    
    # 마스킹 적용
    masked_value = mask.mask(30)
    log_masked_value = log_mask.mask(30)
    
    assert isinstance(masked_value, float)
    assert isinstance(log_masked_value, float)
    assert masked_value != log_masked_value


def test_composite_mask():
    """CompositeMask 테스트"""
    mask = CompositeMask(seed=42)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    assert mask.initialized is True
    assert len(mask.params) == 9  # 9개 파라미터 확인
    
    # 마스킹 적용
    masked_value = mask.mask(30)
    assert isinstance(masked_value, float)
    
    # 도함수 계산
    deriv = mask.derivative(30)
    assert isinstance(deriv, float)
def test_hmac_mask():
    """HMACMask 테스트"""
    mask = HMACMask(seed=42, hmac_key="test_key")
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    assert mask.initialized is True
    
    # 마스킹 적용
    masked_value = mask.mask(30)
    assert isinstance(masked_value, str)
    assert ":" in masked_value  # HMAC과 마스킹 값 구분자 확인
    
    # 동일한 입력에 대한 일관성 확인
    masked_value2 = mask.mask(30)
    assert masked_value == masked_value2
    
    # 다른 키로 마스킹 시 다른 결과 확인
    mask2 = HMACMask(seed=42, hmac_key="different_key")
    mask2.fit(data)
    masked_value3 = mask2.mask(30)
    assert masked_value != masked_value3


def test_dp_noise():
    """차분 프라이버시 노이즈 테스트"""
    # DP 설정
    mask = SinMask(seed=42, dp_epsilon=1.0)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    # 여러 번 마스킹 적용 시 결과 다양성 확인
    masked_values = [mask.mask(30) for _ in range(10)]
    assert len(set(masked_values)) > 1  # 노이즈로 인해 값들이 다름
    
    # 엡실론 값이 클수록 노이즈가 작아짐
    mask_small_noise = SinMask(seed=42, dp_epsilon=10.0)
    mask_small_noise.fit(data)
    
    mask_large_noise = SinMask(seed=42, dp_epsilon=0.1)
    mask_large_noise.fit(data)
    
    # 각 마스크로 여러 번 마스킹
    values_small_noise = [mask_small_noise.mask(30) for _ in range(30)]
    values_large_noise = [mask_large_noise.mask(30) for _ in range(30)]
    
    # 값의 분산 비교
    var_small = np.var(values_small_noise)
    var_large = np.var(values_large_noise)
    
    assert var_large > var_small  # 작은 엡실론(큰 노이즈)이 더 큰 분산을 가짐
