"""
마스킹 알고리즘 기본 인터페이스 클래스 정의
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class IFitMask(ABC):
    """마스킹 알고리즘 기본 인터페이스"""
    
    def __init__(
        self, 
        seed: int = 42, 
        bounds: Optional[Tuple[float, float]] = None, 
        dp_epsilon: Optional[float] = None
    ):
        """
        Args:
            seed: 파라미터 생성 시드
            bounds: 입력값 범위 [min, max]
            dp_epsilon: 차분 프라이버시 엡실론 값 (None이면 적용 안함)
        """
        self.seed = seed
        self.bounds = bounds
        self.dp_epsilon = dp_epsilon
        self.initialized = False
        self.params = {}
    @abstractmethod
    def fit(self, data: Union[List[Union[int, float]], np.ndarray]) -> 'IFitMask':
        """데이터를 기반으로 마스킹 파라미터 초기화

        Args:
            data: 마스킹할 데이터 샘플

        Returns:
            초기화된 self 객체
        """
        pass
    
    @abstractmethod
    def mask(self, value: Union[int, float]) -> Union[str, float]:
        """단일 값을 마스킹

        Args:
            value: 마스킹할 값

        Returns:
            마스킹된 값
        """
        pass
    
    def mask_all(self, values: Union[List[Union[int, float]], np.ndarray]) -> List[Union[str, float]]:
        """여러 값을 한번에 마스킹

        Args:
            values: 마스킹할 값 목록

        Returns:
            마스킹된 값 목록
        """
        return [self.mask(value) for value in values]
    
    @abstractmethod
    def derivative(self, x: float) -> float:
        """마스킹 함수의 도함수 계산

        Args:
            x: 입력값

        Returns:
            도함수 값
        """
        pass
    
    def get_sensitivity(self) -> float:
        """마스킹 함수의 민감도 계산 (도함수의 최대 절대값)

        Returns:
            계산된 민감도 값
        """
        if self.bounds is None:
            raise ValueError("Bounds must be set to calculate sensitivity")
            
        from funcveil.utils import calculate_sensitivity
        return calculate_sensitivity(
            self.derivative, 
            self.bounds[0], 
            self.bounds[1]
        )
        
    def add_dp_noise(self, value: float) -> float:
        """차분 프라이버시 노이즈 추가

        Args:
            value: 원본 값

        Returns:
            노이즈가 추가된 값
        """
        if self.dp_epsilon is None:
            return value
            
        sensitivity = self.get_sensitivity()
        scale = sensitivity / self.dp_epsilon
        
        from funcveil.utils import laplace_noise
        noise = laplace_noise(scale)
        return value + noise
        
    def __call__(self, value: Union[int, float]) -> Union[str, float]:
        """마스킹 함수 호출 간편화

        Args:
            value: 마스킹할 값

        Returns:
            마스킹된 값
        """
        return self.mask(value)
