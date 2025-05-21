"""
지수 및 로그 함수 기반 마스킹 알고리즘 구현
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from funcveil.masking.base import IFitMask
from funcveil.utils import (
    normalize_range, 
    generate_parameters_from_seed,
    set_seed
)


class LogExpMask(IFitMask):
    """지수 및 로그 함수 기반 마스킹 클래스
    
    미분방정식 df/dx = kf(x) 의 해: f(x) = C*e^(kx)
    또는 로그 변환: f(x) = a*log(b*x + c) + d
    """
    
    def __init__(
        self, 
        seed: int = 42, 
        bounds: Optional[Tuple[float, float]] = None,
        dp_epsilon: Optional[float] = None,
        use_log: bool = False,
        round_digits: int = 2
    ):
        """
        Args:
            seed: 파라미터 생성 시드
            bounds: 입력값 범위 [min, max]
            dp_epsilon: 차분 프라이버시 엡실론 값 (None이면 적용 안함)
            use_log: 지수 대신 로그함수 사용 여부
            round_digits: 마스킹 결과 반올림 자릿수
        """
        super().__init__(seed, bounds, dp_epsilon)
        self.use_log = use_log
        self.round_digits = round_digits
    def fit(self, data: Union[List[Union[int, float]], np.ndarray]) -> 'LogExpMask':
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
        
        if self.use_log:
            # 로그 함수 파라미터: f(x) = a*log(b*x + c) + d
            self.params = {
                'a': params[0],  # 로그 스케일
                'b': max(0.1, params[1]),  # 입력 스케일 (양수)
                'c': params[2] + 1,  # 로그 안전성을 위한 오프셋 (항상 > 0)
                'd': params[3]  # y축 이동
            }
        else:
            # 지수 함수 파라미터: f(x) = C*e^(kx)
            self.params = {
                'C': params[0],  # 스케일
                'k': params[1] * 0.2,  # 증가율 (너무 급격하지 않게 조정)
                'shift': params[2]  # x축 이동
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
            
        if self.use_log:
            # 로그 함수 적용: f(x) = a*log(b*x + c) + d
            # 로그 인자는 항상 양수여야 함
            log_input = max(1e-10, self.params['b'] * normalized_x + self.params['c'])
            result = (
                self.params['a'] * 
                np.log(log_input) + 
                self.params['d']
            )
        else:
            # 지수 함수 적용: f(x) = C*e^(k*(x-shift))
            result = (
                self.params['C'] * 
                np.exp(self.params['k'] * (normalized_x - self.params['shift']))
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
        
        if self.use_log:
            # d/dx [a*log(b*x + c) + d] = a*b / (b*x + c)
            log_input = max(1e-10, self.params['b'] * normalized_x + self.params['c'])
            return self.params['a'] * self.params['b'] / log_input
        else:
            # d/dx [C*e^(k*(x-shift))] = C*k*e^(k*(x-shift))
            return (
                self.params['C'] * 
                self.params['k'] * 
                np.exp(self.params['k'] * (normalized_x - self.params['shift']))
            )
