"""
마스킹 함수의 가역성(역변환 가능성) 평가 도구
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from collections import defaultdict

from funcveil.masking.base import IFitMask


def calculate_invertibility_score(
    mask_func: IFitMask, 
    samples: int = 100, 
    bins: int = 10, 
    plot: bool = False
) -> float:
    """마스킹 함수의 역변환 가능성(Invertibility) 점수 계산

    Args:
        mask_func: 마스킹 객체
        samples: 테스트할 샘플 수
        bins: 결과값을 나눌 구간 수
        plot: 결과 시각화 여부

    Returns:
        역변환 가능성 점수 (1.0이면 일대일 대응, 값이 클수록 역변환 어려움)
    """
    if not mask_func.initialized:
        raise ValueError("Mask function not initialized. Call fit() first.")
        
    if mask_func.bounds is None:
        raise ValueError("Mask function must have bounds defined")
    
    # 테스트 데이터 생성
    x_vals = np.linspace(mask_func.bounds[0], mask_func.bounds[1], samples)
    
    # 마스킹 적용
    y_vals = [float(str(mask_func(x)).split(':')[-1]) if ':' in str(mask_func(x)) 
              else float(mask_func(x)) for x in x_vals]
    
    # 결과값 기준으로 binning
    y_min, y_max = min(y_vals), max(y_vals)
    bin_edges = np.linspace(y_min, y_max, bins + 1)
    
    # 각 bin에 속하는 x값 개수 계산
    bin_counts = defaultdict(int)
    for y in y_vals:
        for i in range(bins):
            if bin_edges[i] <= y < bin_edges[i + 1]:
                bin_counts[i] += 1
                break
        if y == y_max:  # 마지막 값 처리
            bin_counts[bins - 1] += 1
    
    # 평균 x 개수가 1이면 일대일 대응, 크면 다대일 대응
    non_empty_bins = len([b for b in bin_counts.values() if b > 0])
    avg_xs_per_bin = samples / non_empty_bins if non_empty_bins > 0 else float('inf')
    # 시각화
    if plot:
        plt.figure(figsize=(10, 6))
        
        # 함수 그래프
        plt.subplot(1, 2, 1)
        plt.scatter(x_vals, y_vals, alpha=0.6)
        plt.title(f'Mask Function')
        plt.xlabel('Input Value')
        plt.ylabel('Masked Value')
        plt.grid(True)
        
        # 역변환 가능성 분포
        plt.subplot(1, 2, 2)
        bin_values = list(bin_counts.values())
        bin_indices = list(bin_counts.keys())
        plt.bar([str(i) for i in bin_indices], bin_values)
        plt.title(f'Invertibility Score: {avg_xs_per_bin:.2f}')
        plt.xlabel('Output Value Bins')
        plt.ylabel('Count of Input Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return avg_xs_per_bin


def visualize_mask_function(
    mask_func: IFitMask, 
    samples: int = 100,
    with_derivative: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """마스킹 함수와 도함수 시각화

    Args:
        mask_func: 마스킹 객체
        samples: 시각화할 샘플 수
        with_derivative: 도함수 포함 여부
        figsize: 그래프 크기
    """
    if not mask_func.initialized:
        raise ValueError("Mask function not initialized. Call fit() first.")
        
    if mask_func.bounds is None:
        raise ValueError("Mask function must have bounds defined")
    
    # 테스트 데이터 생성
    x_vals = np.linspace(mask_func.bounds[0], mask_func.bounds[1], samples)
    
    # 마스킹 적용
    y_vals = [float(str(mask_func(x)).split(':')[-1]) if ':' in str(mask_func(x)) 
              else float(mask_func(x)) for x in x_vals]
    plt.figure(figsize=figsize)
    
    if with_derivative:
        # 도함수 계산
        dy_vals = [mask_func.derivative(x) for x in x_vals]
        
        # 함수 그래프
        plt.subplot(2, 1, 1)
    
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.title(f'Mask Function: {mask_func.__class__.__name__}')
    plt.xlabel('Input Value')
    plt.ylabel('Masked Value')
    plt.grid(True)
    
    if with_derivative:
        # 도함수 그래프
        plt.subplot(2, 1, 2)
        plt.plot(x_vals, dy_vals, 'r-', linewidth=2)
        plt.title('Derivative (Sensitivity)')
        plt.xlabel('Input Value')
        plt.ylabel('Derivative Value')
        plt.grid(True)
        
        # 민감도 표시
        sensitivity = mask_func.get_sensitivity()
        plt.axhline(y=sensitivity, color='g', linestyle='--', label=f'Max Sensitivity: {sensitivity:.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
