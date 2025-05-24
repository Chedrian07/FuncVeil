"""
마스킹 함수의 가역성(역변환 가능성) 평가 도구
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from funcveil.masking.base import IFitMask


def calculate_collision_factor(inputs: List[float], outputs: List[float]) -> float:
    """충돌 계수 계산 - 동일한 출력값을 생성하는 입력값의 개수를 측정
    
    Args:
        inputs: 입력값 리스트
        outputs: 출력값 리스트
        
    Returns:
        충돌 계수 (0~1, 1에 가까울수록 가역성 높음)
    """
    output_to_inputs = defaultdict(list)
    for x, y in zip(inputs, outputs):
        output_to_inputs[y].append(x)
    
    collision_count = 0
    for y, x_list in output_to_inputs.items():
        if len(x_list) > 1:
            collision_count += len(x_list) - 1
    
    # 충돌 계수를 0~1 범위로 정규화 (1이 완전한 일대일 대응)
    invertibility_score = 1.0 / (1.0 + collision_count / len(inputs))
    return invertibility_score


def calculate_information_preservation_rate(inputs: List[float], outputs: List[float]) -> float:
    """정보 보존율 계산 - 고유한 출력값의 비율 측정
    
    Args:
        inputs: 입력값 리스트
        outputs: 출력값 리스트
        
    Returns:
        정보 보존율 (0~1, 1에 가까울수록 정보 보존 우수)
    """
    unique_inputs = len(set(inputs))
    unique_outputs = len(set(outputs))
    
    if unique_inputs == 0:
        return 0.0
    
    preservation_rate = unique_outputs / unique_inputs
    return min(preservation_rate, 1.0)  # 최대값을 1로 제한


def calculate_inverse_accuracy(
    mask_func: IFitMask, 
    inputs: List[float], 
    outputs: List[float],
    tolerance: float = 1e-6
) -> float:
    """역변환 정확도 계산 - 역함수가 존재하는 경우 원본 복원 정확도 측정
    
    Args:
        mask_func: 마스킹 객체
        inputs: 입력값 리스트
        outputs: 출력값 리스트
        tolerance: 허용 오차
        
    Returns:
        역변환 정확도 (0~1, 1에 가까울수록 역변환 정확)
    """
    # 역함수가 구현되어 있지 않은 경우 충돌 기반 추정
    output_to_inputs = defaultdict(list)
    for x, y in zip(inputs, outputs):
        output_to_inputs[y].append(x)
    
    correct_inversions = 0
    total_inversions = 0
    
    for y, x_list in output_to_inputs.items():
        if len(x_list) == 1:
            # 유일한 입력값을 가진 경우 완벽한 역변환 가능
            correct_inversions += 1
            total_inversions += 1
        else:
            # 여러 입력값을 가진 경우 역변환 불가능
            total_inversions += len(x_list)
    
    if total_inversions == 0:
        return 0.0
    
    return correct_inversions / total_inversions


def calculate_invertibility_score(
    mask_func: IFitMask, 
    samples: int = 100, 
    bins: int = 10, 
    plot: bool = False,
    return_dict: bool = False
) -> Union[float, Dict[str, float]]:
    """마스킹 함수의 가역성 종합 평가

    Args:
        mask_func: 마스킹 객체
        samples: 테스트할 샘플 수
        bins: 결과값을 나눌 구간 수 (시각화용)
        plot: 결과 시각화 여부
        return_dict: True일 경우 모든 지표를 딕셔너리로 반환, False일 경우 종합 점수만 반환

    Returns:
        return_dict가 True면 가역성 평가 지표들의 딕셔너리, False면 종합 점수 (float)
    """
    if not mask_func.initialized:
        raise ValueError("Mask function not initialized. Call fit() first.")
        
    if mask_func.bounds is None:
        raise ValueError("Mask function must have bounds defined")
    
    # 테스트 데이터 생성
    x_vals = np.linspace(mask_func.bounds[0], mask_func.bounds[1], samples)
    
    # 마스킹 적용
    y_vals = []
    for x in x_vals:
        masked_val = mask_func(x)
        # 다양한 반환값 형식 처리
        if isinstance(masked_val, (int, float, np.integer, np.floating)):
            y_vals.append(float(masked_val))
        elif isinstance(masked_val, np.ndarray):
            # numpy 배열인 경우 첫 번째 요소 사용
            y_vals.append(float(masked_val.flatten()[0]))
        elif isinstance(masked_val, str):
            # 문자열 처리
            if ':' in masked_val:
                # 'key:value' 형식
                y_vals.append(float(masked_val.split(':')[-1]))
            elif masked_val.startswith('[') and masked_val.endswith(']'):
                # '[value]' 형식
                cleaned = masked_val.strip('[]')
                y_vals.append(float(cleaned))
            else:
                y_vals.append(float(masked_val))
        else:
            # 기타 타입은 float 변환 시도
            y_vals.append(float(masked_val))
    
    # 가역성 지표 계산
    collision_factor = calculate_collision_factor(x_vals.tolist(), y_vals)
    preservation_rate = calculate_information_preservation_rate(x_vals.tolist(), y_vals)
    inverse_accuracy = calculate_inverse_accuracy(mask_func, x_vals.tolist(), y_vals)
    
    # 종합 점수 계산 (가중평균)
    overall_score = (
        collision_factor * 0.4 + 
        preservation_rate * 0.3 + 
        inverse_accuracy * 0.3
    ) * 10  # 0-10 스케일로 변환
    
    results = {
        'overall_score': overall_score,
        'collision_factor': collision_factor,
        'information_preservation_rate': preservation_rate,
        'inverse_accuracy': inverse_accuracy
    }
    
    # 시각화
    if plot:
        visualize_invertibility_analysis(mask_func, x_vals, y_vals, results, bins)
    
    # 반환값 결정
    if return_dict:
        return results
    else:
        return overall_score


def visualize_invertibility_analysis(
    mask_func: IFitMask,
    x_vals: np.ndarray,
    y_vals: List[float],
    results: Dict[str, float],
    bins: int = 10
) -> None:
    """가역성 분석 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 함수 그래프
    ax = axes[0, 0]
    ax.scatter(x_vals, y_vals, alpha=0.6, s=10)
    ax.set_title(f'{mask_func.__class__.__name__} Function Mapping')
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Masked Value')
    ax.grid(True, alpha=0.3)
    
    # 2. 출력값 분포
    ax = axes[0, 1]
    y_counts = Counter(y_vals)
    unique_y_vals = sorted(y_counts.keys())
    counts = [y_counts[y] for y in unique_y_vals]
    
    if len(unique_y_vals) > 50:
        # 많은 경우 히스토그램으로 표시
        ax.hist(y_vals, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Frequency')
    else:
        # 적은 경우 막대그래프로 표시
        ax.bar(range(len(unique_y_vals)), counts, alpha=0.7)
        ax.set_ylabel('Input Count per Output')
        ax.set_xticks([])
    
    ax.set_title('Output Value Distribution')
    ax.set_xlabel('Output Values')
    
    # 3. 충돌 분석
    ax = axes[1, 0]
    collision_data = defaultdict(int)
    for count in y_counts.values():
        collision_data[count] += 1
    
    collision_counts = sorted(collision_data.keys())
    collision_freqs = [collision_data[c] for c in collision_counts]
    
    ax.bar(collision_counts, collision_freqs, alpha=0.7, color='red')
    ax.set_title('Collision Analysis')
    ax.set_xlabel('Number of Inputs per Output')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # 4. 가역성 점수 요약
    ax = axes[1, 1]
    metrics = [
        ('Overall Score', results['overall_score']),
        ('Collision Factor', results['collision_factor'] * 10),
        ('Info. Preservation', results['information_preservation_rate'] * 10),
        ('Inverse Accuracy', results['inverse_accuracy'] * 10)
    ]
    
    names, scores = zip(*metrics)
    colors = ['darkblue', 'green', 'orange', 'purple']
    bars = ax.bar(names, scores, color=colors, alpha=0.7)
    
    # 점수 표시
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom')
    
    ax.set_title('Invertibility Scores (0-10 scale)')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Invertibility Analysis: {mask_func.__class__.__name__}', fontsize=14)
    plt.tight_layout()
    plt.show()


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
