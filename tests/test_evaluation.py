"""
평가 도구 테스트
"""

import pytest
import numpy as np
from funcveil.masking.sin import SinMask
from funcveil.evaluation.reversibility import calculate_invertibility_score, visualize_mask_function
from funcveil.evaluation.dp_sensitivity import evaluate_dp_sensitivity, compare_dp_impact


def test_invertibility_score():
    """역변환 가능성 점수 계산 테스트"""
    # 일반 마스크
    mask = SinMask(seed=42)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    score = calculate_invertibility_score(mask, plot=False)
    assert isinstance(score, float)
    assert score > 0
    
    # 단조 마스크 (역변환 가능)
    monotonic_mask = SinMask(seed=42, monotonic=True)
    monotonic_mask.fit(data)
    
    monotonic_score = calculate_invertibility_score(monotonic_mask, plot=False)
    assert isinstance(monotonic_score, float)
    assert monotonic_score > 0
    
    # 단조 마스크는 일반 마스크보다 역변환 가능성이 높아야 함 (점수는 낮음)
    assert monotonic_score <= score


def test_dp_sensitivity_evaluation():
    """DP 민감도 평가 테스트"""
    mask = SinMask(seed=42)
    data = np.array([10, 20, 30, 40, 50])
    mask.fit(data)
    
    # 간단한 테스트를 위해 엡실론 값 하나만 사용
    eps_values = [1.0]
    result = evaluate_dp_sensitivity(mask, epsilon_values=eps_values, plot=False)
    
    assert 'sensitivity' in result
    assert isinstance(result['sensitivity'], float)
    assert 'epsilon_stats' in result
    assert 1.0 in result['epsilon_stats']
    assert 'scale' in result['epsilon_stats'][1.0]
