"""
사인 함수 기반 마스킹 알고리즘 구현
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from funcveil.masking.base import IFitMask
from funcveil.utils import (
    normalize_range, 
    generate_parameters_from_seed,
    set_seed
)


class SinMask(IFitMask):
    """사인 함수 기반 마스킹 클래스
    
    f(x) = A * sin(ω * ((x - L) / (U - L)) + φ) + B
    
    - x ∈ [L, U]: 정규화된 입력값 범위
    - A, ω, φ, B: 시드 기반 파라미터
    """
    
    def __init__(
        self, 
        seed: int = 42, 
        bounds: Optional[Tuple[float, float]] = None,
        dp_epsilon: Optional[float] = None,
        monotonic: bool = False,
        round_digits: int = 2
    ):
        """
        Args:
            seed: 파라미터 생성 시드
            bounds: 입력값 범위 [min, max]
            dp_epsilon: 차분 프라이버시 엡실론 값 (None이면 적용 안함)
            monotonic: 단조 증가 구간 설정 여부
            round_digits: 마스킹 결과 반올림 자릿수
        """
        super().__init__(seed, bounds, dp_epsilon)
        self.monotonic = monotonic
        self.round_digits = round_digits
    def fit(self, data: Union[List[Union[int, float]], np.ndarray]) -> 'SinMask':
        """데이터를 기반으로 마스킹 파라미터 초기화

        Args:
            data: 마스킹할 데이터 샘플

        Returns:
            초기화된 self 객체
        """
        # 데이터 범위 계산
        if self.bounds is None:
            self.bounds = (min(data), max(data))
        
        # 파라미터 생성
        params = generate_parameters_from_seed(self.seed, 4)
        
        # 파라미터 설정
        self.params = {
            'A': params[0],  # 진폭
            'omega': params[1],  # 각속도
            'phi': params[2],  # 위상
            'B': params[3]  # y축 이동
        }
        
        # 단조성 조건 적용 (ω(U-L) ≤ π)
        if self.monotonic:
            range_diff = self.bounds[1] - self.bounds[0]
            self.params['omega'] = min(self.params['omega'], np.pi / range_diff)
        
        self.initialized = True
        return self
    
    def _normalize_input(self, value: Union[int, float]) -> float:
        """입력값을 [0, 1] 범위로 정규화

        Args:
            value: 정규화할 값

        Returns:
            정규화된 값
        """
        if self.bounds is None:
            raise ValueError("Bounds must be set before normalization")
            
        return normalize_range(value, self.bounds[0], self.bounds[1], 0, 1)
    def _calculate_mask(self, normalized_x: float) -> float:
        """정규화된 값에 대해 마스킹 함수 계산

        Args:
            normalized_x: 정규화된 입력값 (0-1 범위)

        Returns:
            마스킹된 값
        """
        if not self.initialized:
            raise ValueError("Mask parameters not initialized. Call fit() first.")
            
        # 사인 함수 적용
        result = (
            self.params['A'] * 
            np.sin(self.params['omega'] * normalized_x + self.params['phi']) + 
            self.params['B']
        )
        
        return result
    
    def mask(self, value: Union[int, float]) -> float:
        """단일 값을 마스킹

        Args:
            value: 마스킹할 값

        Returns:
            마스킹된 값
        """
        normalized = self._normalize_input(value)
        masked_value = self._calculate_mask(normalized)
        
        # 차분 프라이버시 노이즈 추가 (설정된 경우)
        if self.dp_epsilon is not None:
            masked_value = self.add_dp_noise(masked_value)
            
        # 결과 반올림 - numpy 배열 처리 추가
        if isinstance(masked_value, np.ndarray):
            return np.round(masked_value, self.round_digits)
        else:
            return round(masked_value, self.round_digits)
    
    def derivative(self, x: float) -> float:
        """마스킹 함수의 도함수 계산

        Args:
            x: 입력값

        Returns:
            도함수 값
        """
        if not self.initialized:
            raise ValueError("Mask parameters not initialized. Call fit() first.")
            
        normalized_x = self._normalize_input(x)
        
        # d/dx [A * sin(ω*x + φ) + B] = A * ω * cos(ω*x + φ)
        return (
            self.params['A'] * 
            self.params['omega'] * 
            np.cos(self.params['omega'] * normalized_x + self.params['phi'])
        )
