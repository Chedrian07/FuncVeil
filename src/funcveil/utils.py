"""
유틸리티 함수 모음

주요 기능:
- 랜덤 시드 관리
- 값 범위 정규화
- 환경설정 로드
"""

import os
import numpy as np
import hashlib
import hmac
import random
import binascii
from typing import Union, List, Tuple, Dict, Any, Optional


def set_seed(seed: int = 42) -> None:
    """난수 생성 시드 설정

    Args:
        seed: 설정할 시드값
    """
    random.seed(seed)
    np.random.seed(seed)


def normalize_range(
    value: Union[int, float], 
    original_min: Union[int, float], 
    original_max: Union[int, float], 
    target_min: Union[int, float] = 0, 
    target_max: Union[int, float] = 1
) -> float:
    """값 범위 정규화 함수

    Args:
        value: 정규화할 원래 값
        original_min: 원래 범위의 최소값
        original_max: 원래 범위의 최대값
        target_min: 변환 대상 범위의 최소값
        target_max: 변환 대상 범위의 최대값

    Returns:
        정규화된 값
    """
    if original_max == original_min:
        return target_min
    
    normalized = (value - original_min) / (original_max - original_min)
    return normalized * (target_max - target_min) + target_min


def generate_parameters_from_seed(seed: int, param_count: int = 4) -> List[float]:
    """시드값으로부터 마스킹 파라미터 생성

    Args:
        seed: 시드값
        param_count: 생성할 파라미터 개수

    Returns:
        생성된 파라미터 리스트
    """
    set_seed(seed)
    # 안정적인 구조를 위한 파라미터 범위 지정
    param_ranges = [
        (1.0, 5.0),    # A: 진폭
        (0.5, 2.0),    # ω: 각속도
        (0.0, 2*np.pi), # φ: 위상
        (0.0, 10.0)    # B: y축 이동
    ]
    
    params = []
    for i in range(param_count):
        min_val, max_val = param_ranges[i % len(param_ranges)]
        params.append(min_val + (max_val - min_val) * random.random())
        
    return params


def create_hmac(data: Union[str, int, float], key: str) -> str:
    """HMAC 해시 생성

    Args:
        data: 해시할 데이터
        key: HMAC 키

    Returns:
        HMAC 해시값 (16진수 문자열)
    """
    data_str = str(data).encode('utf-8')
    key_bytes = key.encode('utf-8')
    
    hmac_obj = hmac.new(key_bytes, data_str, hashlib.sha256)
    return hmac_obj.hexdigest()


def truncate_hmac(hmac_hex: str, length: int = 8) -> str:
    """HMAC 해시값을 지정된 길이로 자름

    Args:
        hmac_hex: HMAC 해시값
        length: 원하는 문자 길이

    Returns:
        잘린 해시값
    """
    return hmac_hex[:length]


def laplace_noise(scale: float, size: int = 1) -> Union[float, np.ndarray]:
    """차분 프라이버시를 위한 라플라스 노이즈 생성

    Args:
        scale: 노이즈 스케일 (b)
        size: 생성할 노이즈 개수

    Returns:
        생성된 라플라스 노이즈
    """
    return np.random.laplace(0, scale, size)


def calculate_sensitivity(derivative_func, range_min: float, range_max: float, samples: int = 100) -> float:
    """함수의 민감도 계산 (도함수의 최대 절대값)

    Args:
        derivative_func: 도함수
        range_min: 범위 최소값
        range_max: 범위 최대값
        samples: 샘플링 개수

    Returns:
        계산된 민감도 값
    """
    x_values = np.linspace(range_min, range_max, samples)
    derivatives = [abs(derivative_func(x)) for x in x_values]
    return max(derivatives)
