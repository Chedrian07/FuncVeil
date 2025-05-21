"""
복합 함수 (sin + cos) 기반 마스킹 알고리즘 구현
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from funcveil.masking.base import IFitMask
from funcveil.utils import (
    normalize_range, 
    generate_parameters_from_seed,
    set_seed
)


class CompositeMask(IFitMask):
    """sin과 cos 함수 조합 기반 마스킹 클래스
    
    f(x) = A1 * sin(ω1 * x + φ1) + B1 + A2 * cos(ω2 * x + φ2) + B2 + C
    """
    
    def __init__(
        self, 
        seed: int = 42, 
        bounds: Optional[Tuple[float, float]] = None,
        dp_epsilon: Optional[float] = None,
        round_digits: int = 2
    ):
        """
        Args:
            seed: 파라미터 생성 시드
            bounds: 입력값 범위 [min, max]
            dp_epsilon: 차분 프라이버시 엡실론 값 (None이면 적용 안함)
            round_digits: 마스킹 결과 반올림 자릿수
        """
        super().__init__(seed, bounds, dp_epsilon)
        self.round_digits = round_digits
    def fit(self, data: Union[List[Union[int, float]], np.ndarray]) -> 'CompositeMask':
        """데이터를 기반으로 마스킹 파라미터 초기화

        Args:
            data: 마스킹할 데이터 샘플

        Returns:
            초기화된 self 객체
        """
        # 데이터 범위 계산
        if self.bounds is None:
            self.bounds = (min(data), max(data))
        
        # 파라미터 생성 (sin, cos 각각 4개 파라미터 + 추가 상수 1개)
        params = generate_parameters_from_seed(self.seed, 9)
        
        # 파라미터 설정 (sin 부분)
        self.params = {
            'A1': params[0],      # sin 진폭
            'omega1': params[1],  # sin 각속도
            'phi1': params[2],    # sin 위상
            'B1': params[3],      # sin y축 이동
            
            # cos 부분
            'A2': params[4],      # cos 진폭
            'omega2': params[5],  # cos 각속도
            'phi2': params[6],    # cos 위상
            'B2': params[7],      # cos y축 이동
            
            # 추가 상수
            'C': params[8]        # 전체 y축 이동
        }
        
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
            
        # sin 부분 계산
        sin_part = (
            self.params['A1'] * 
            np.sin(self.params['omega1'] * normalized_x + self.params['phi1']) + 
            self.params['B1']
        )
        
        # cos 부분 계산
        cos_part = (
            self.params['A2'] * 
            np.cos(self.params['omega2'] * normalized_x + self.params['phi2']) + 
            self.params['B2']
        )
        
        # 전체 함수 계산
        result = sin_part + cos_part + self.params['C']
        
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
        
        # sin 부분 도함수: d/dx [A1 * sin(ω1*x + φ1) + B1] = A1 * ω1 * cos(ω1*x + φ1)
        sin_deriv = (
            self.params['A1'] * 
            self.params['omega1'] * 
            np.cos(self.params['omega1'] * normalized_x + self.params['phi1'])
        )
        
        # cos 부분 도함수: d/dx [A2 * cos(ω2*x + φ2) + B2] = -A2 * ω2 * sin(ω2*x + φ2)
        cos_deriv = (
            -self.params['A2'] * 
            self.params['omega2'] * 
            np.sin(self.params['omega2'] * normalized_x + self.params['phi2'])
        )
        
        # 전체 도함수
        return sin_deriv + cos_deriv
