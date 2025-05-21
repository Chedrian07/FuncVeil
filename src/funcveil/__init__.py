"""
FuncVeil: 미적분 기반 비가역 개인정보 마스킹 시스템
===============================================

미적분 개념을 활용한 복호화 불가 마스킹 알고리즘을 통해 
개인정보를 안전하게 보호하면서도 통계적 분석이 가능한 데이터로 변환하는 라이브러리

주요 기능:
- 연속함수 기반 마스킹 (sin, cos)
- 지수/로그 기반 마스킹
- 복합 함수 마스킹
- HMAC 해시 결합
- 차분 프라이버시(DP) 노이즈 적용
- 마스킹 결과 평가 도구

"""

__version__ = "0.1.0"
__author__ = "FuncVeil Team"

from funcveil.masking.base import IFitMask
from funcveil.masking.sin import SinMask
from funcveil.masking.logexp import LogExpMask
from funcveil.masking.composite import CompositeMask
from funcveil.masking.hmac_mask import HMACMask

__all__ = [
    "IFitMask", 
    "SinMask", 
    "LogExpMask", 
    "CompositeMask", 
    "HMACMask",
]
