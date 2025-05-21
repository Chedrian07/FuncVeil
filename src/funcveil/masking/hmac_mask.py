"""
HMAC 결합 마스킹 알고리즘 구현
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from funcveil.masking.base import IFitMask
from funcveil.masking.sin import SinMask
from funcveil.utils import (
    normalize_range, 
    generate_parameters_from_seed,
    create_hmac,
    truncate_hmac
)


class HMACMask(IFitMask):
    """HMAC 해시와 마스킹 함수를 결합한 마스킹 클래스
    
    f(x) = HMAC_key(x) || round(A * sin(ω*x + φ) + B, n)
    """
    
    def __init__(
        self, 
        seed: int = 42, 
        bounds: Optional[Tuple[float, float]] = None,
        dp_epsilon: Optional[float] = None,
        hmac_key: str = "default_key",
        hmac_length: int = 8,
        base_mask: Optional[IFitMask] = None,
        round_digits: int = 2
    ):
        """
        Args:
            seed: 파라미터 생성 시드
            bounds: 입력값 범위 [min, max]
            dp_epsilon: 차분 프라이버시 엡실론 값 (None이면 적용 안함)
            hmac_key: HMAC 생성에 사용할 키
            hmac_length: HMAC 해시 잘라낼 길이
            base_mask: 기본 마스킹 함수 (기본값: SinMask)
            round_digits: 마스킹 결과 반올림 자릿수
        """
        super().__init__(seed, bounds, dp_epsilon)
        self.hmac_key = hmac_key
        self.hmac_length = hmac_length
        self.round_digits = round_digits
        
        # 기본 마스킹 함수 설정
        if base_mask is None:
            self.base_mask = SinMask(seed, bounds, dp_epsilon, round_digits=round_digits)
        else:
            self.base_mask = base_mask
    def fit(self, data: Union[List[Union[int, float]], np.ndarray]) -> 'HMACMask':
        """데이터를 기반으로 마스킹 파라미터 초기화

        Args:
            data: 마스킹할 데이터 샘플

        Returns:
            초기화된 self 객체
        """
        # 기본 마스킹 함수 초기화
        self.base_mask.fit(data)
        
        # 자체 초기화 완료 설정
        self.initialized = True
        return self
    
    def mask(self, value: Union[int, float]) -> str:
        """단일 값을 마스킹

        Args:
            value: 마스킹할 값

        Returns:
            마스킹된 값 (HMAC 해시 + 마스킹된 숫자)
        """
        if not self.initialized:
            raise ValueError("Mask parameters not initialized. Call fit() first.")
            
        # 기본 마스킹 적용
        masked_value = self.base_mask.mask(value)
        
        # HMAC 해시 생성 및 잘라내기
        hmac_hash = create_hmac(value, self.hmac_key)
        truncated_hmac = truncate_hmac(hmac_hash, self.hmac_length)
        
        # 결합: hash || masked
        return f"{truncated_hmac}:{masked_value}"
    
    def derivative(self, x: float) -> float:
        """마스킹 함수의 도함수 계산 (기본 마스킹 함수에 위임)

        Args:
            x: 입력값

        Returns:
            도함수 값
        """
        return self.base_mask.derivative(x)
