"""
차분 프라이버시(DP)와 민감도 분석 도구
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

from funcveil.masking.base import IFitMask
from funcveil.utils import laplace_noise


def evaluate_dp_sensitivity(
    mask_func: IFitMask,
    epsilon_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    samples: int = 1000,
    repetitions: int = 10,
    plot: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> dict:
    """차분 프라이버시 민감도 평가

    Args:
        mask_func: 마스킹 객체
        epsilon_values: 테스트할 ε 값 목록
        samples: 테스트할 샘플 수
        repetitions: 각 샘플당 반복 횟수
        plot: 결과 시각화 여부
        figsize: 그래프 크기

    Returns:
        ε별 노이즈 통계 정보
    """
    if not mask_func.initialized:
        raise ValueError("Mask function not initialized. Call fit() first.")
        
    if mask_func.bounds is None:
        raise ValueError("Mask function must have bounds defined")
    
    # 민감도 계산
    sensitivity = mask_func.get_sensitivity()
    
    # 결과 저장 딕셔너리
    results = {
        'sensitivity': sensitivity,
        'epsilon_stats': {}
    }
    
    # 테스트 데이터 생성
    x_vals = np.linspace(mask_func.bounds[0], mask_func.bounds[1], samples)
    
    # 각 ε 값에 대해 노이즈 생성 및 평가
    for eps in epsilon_values:
        scale = sensitivity / eps
        
        # 모든 샘플에 대해 반복 측정
        noise_samples = []
        
        for x in x_vals:
            # 마스킹 기본값 (노이즈 없음)
            base_value = float(str(mask_func(x)).split(':')[-1]) if ':' in str(mask_func(x)) else float(mask_func(x))
            
            # 여러 번 노이즈 추가해서 변동성 측정
            for _ in range(repetitions):
                noise = laplace_noise(scale)
                noisy_value = base_value + noise
                noise_samples.append(noise)
        # 노이즈 통계 계산
        noise_array = np.array(noise_samples)
        noise_stats = {
            'mean': float(np.mean(noise_array)),
            'std': float(np.std(noise_array)),
            'min': float(np.min(noise_array)),
            'max': float(np.max(noise_array)),
            'abs_mean': float(np.mean(np.abs(noise_array))),
            'scale': float(scale)
        }
        
        results['epsilon_stats'][eps] = noise_stats
    
    # 결과 시각화
    if plot:
        plt.figure(figsize=figsize)
        
        # 노이즈 분포 비교
        plt.subplot(2, 2, 1)
        for eps in epsilon_values:
            scale = sensitivity / eps
            x = np.linspace(-3*scale, 3*scale, 1000)
            y = np.exp(-np.abs(x) / scale) / (2 * scale)  # 라플라스 분포
            plt.plot(x, y, label=f'ε = {eps}')
        
        plt.title('Laplace Noise Distribution by ε')
        plt.xlabel('Noise Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        
        # 평균 절대 노이즈 크기
        plt.subplot(2, 2, 2)
        eps_values = list(results['epsilon_stats'].keys())
        abs_means = [results['epsilon_stats'][eps]['abs_mean'] for eps in eps_values]
        
        plt.plot(eps_values, abs_means, 'o-', linewidth=2)
        plt.title('Mean Absolute Noise vs ε')
        plt.xlabel('ε Value')
        plt.ylabel('Mean Absolute Noise')
        plt.grid(True)
        
        # 표준편차 비교
        plt.subplot(2, 2, 3)
        stds = [results['epsilon_stats'][eps]['std'] for eps in eps_values]
        
        plt.plot(eps_values, stds, 'o-', linewidth=2)
        plt.title('Noise Standard Deviation vs ε')
        plt.xlabel('ε Value')
        plt.ylabel('Standard Deviation')
        plt.grid(True)
        # 민감도와 스케일 관계
        plt.subplot(2, 2, 4)
        scales = [results['epsilon_stats'][eps]['scale'] for eps in eps_values]
        
        plt.plot(eps_values, scales, 'o-', linewidth=2)
        plt.title(f'Scale (b = Δf/ε) vs ε (Δf = {sensitivity:.2f})')
        plt.xlabel('ε Value')
        plt.ylabel('Scale (b)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return results


def compare_dp_impact(
    mask_func: IFitMask,
    epsilon_values: List[float] = [0.1, 1.0, 10.0],
    samples: int = 100,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """DP 적용 여부에 따른 마스킹 결과 비교 시각화

    Args:
        mask_func: 마스킹 객체
        epsilon_values: 테스트할 ε 값 목록
        samples: 테스트할 샘플 수
        figsize: 그래프 크기
    """
    if not mask_func.initialized:
        raise ValueError("Mask function not initialized. Call fit() first.")
        
    if mask_func.bounds is None:
        raise ValueError("Mask function must have bounds defined")
    
    # 테스트 데이터 생성
    x_vals = np.linspace(mask_func.bounds[0], mask_func.bounds[1], samples)
    
    # 기본 마스킹 결과 (노이즈 없음)
    base_y_vals = [float(str(mask_func(x)).split(':')[-1]) if ':' in str(mask_func(x)) 
                  else float(mask_func(x)) for x in x_vals]
    plt.figure(figsize=figsize)
    
    # 기본 마스킹 결과 플롯
    plt.plot(x_vals, base_y_vals, 'k-', linewidth=2, label='No Noise')
    
    # 다양한 ε값에 대한 노이즈 추가 결과
    for eps in epsilon_values:
        sensitivity = mask_func.get_sensitivity()
        scale = sensitivity / eps
        
        # 노이즈 추가
        noisy_y_vals = [base_y + laplace_noise(scale) for base_y in base_y_vals]
        
        plt.plot(x_vals, noisy_y_vals, 'o-', alpha=0.6, markersize=3, 
                 label=f'ε = {eps} (scale = {scale:.2f})')
    
    plt.title('Impact of Differential Privacy Noise on Masking')
    plt.xlabel('Input Value')
    plt.ylabel('Masked Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def quantify_utility_privacy_tradeoff(
    mask_func: IFitMask,
    epsilon_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    samples: int = 1000,
    repetitions: int = 10,
    plot: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> dict:
    """유용성과 개인정보 보호 간 균형점 정량화

    Args:
        mask_func: 마스킹 객체
        epsilon_values: 테스트할 ε 값 목록
        samples: 테스트할 샘플 수
        repetitions: 각 샘플당 반복 횟수
        plot: 결과 시각화 여부
        figsize: 그래프 크기

    Returns:
        ε별 유용성 지표
    """
    # 민감도 평가 결과
    dp_stats = evaluate_dp_sensitivity(
        mask_func, 
        epsilon_values=epsilon_values,
        samples=samples,
        repetitions=repetitions,
        plot=False
    )
    
    sensitivity = dp_stats['sensitivity']
    # 유용성 지표 계산 (신호 대 잡음 비율)
    utility_metrics = {}
    
    # 기본 마스킹 결과에서 유용한 신호 범위 측정
    x_vals = np.linspace(mask_func.bounds[0], mask_func.bounds[1], samples)
    base_y_vals = [float(str(mask_func(x)).split(':')[-1]) if ':' in str(mask_func(x)) 
                  else float(mask_func(x)) for x in x_vals]
    
    signal_range = max(base_y_vals) - min(base_y_vals)
    
    for eps in epsilon_values:
        noise_stats = dp_stats['epsilon_stats'][eps]
        
        # 신호 대 노이즈 비율 (SNR)
        snr = signal_range / (2 * noise_stats['std'])
        
        # 정규화된 오차율 (값이 클수록 유용성 감소)
        error_rate = noise_stats['abs_mean'] / signal_range
        
        utility_metrics[eps] = {
            'signal_to_noise_ratio': snr,
            'normalized_error_rate': error_rate,
            'privacy_score': 1 / eps,  # ε이 작을수록 프라이버시 보호 강화
            'utility_score': snr,  # SNR이 클수록 유용성 높음
            'privacy_utility_product': (1 / eps) * snr  # 균형점 지표
        }
    
    # 시각화
    if plot:
        plt.figure(figsize=figsize)
        
        eps_values = list(utility_metrics.keys())
        
        # SNR 및 오차율
        plt.subplot(1, 2, 1)
        snr_values = [utility_metrics[eps]['signal_to_noise_ratio'] for eps in eps_values]
        error_values = [utility_metrics[eps]['normalized_error_rate'] for eps in eps_values]
        
        plt.plot(eps_values, snr_values, 'bo-', linewidth=2, label='Signal-to-Noise Ratio')
        plt.plot(eps_values, error_values, 'ro-', linewidth=2, label='Normalized Error Rate')
        
        plt.title('Utility Metrics vs ε')
        plt.xlabel('ε Value')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        # 프라이버시-유용성 균형 점수
        plt.subplot(1, 2, 2)
        privacy_scores = [utility_metrics[eps]['privacy_score'] for eps in eps_values]
        utility_scores = [utility_metrics[eps]['utility_score'] for eps in eps_values]
        product_scores = [utility_metrics[eps]['privacy_utility_product'] for eps in eps_values]
        
        plt.plot(eps_values, privacy_scores, 'go-', linewidth=2, label='Privacy Score (1/ε)')
        plt.plot(eps_values, utility_scores, 'bo-', linewidth=2, label='Utility Score (SNR)')
        plt.plot(eps_values, product_scores, 'mo-', linewidth=2, label='Privacy-Utility Product')
        
        plt.title('Privacy-Utility Tradeoff')
        plt.xlabel('ε Value')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        plt.tight_layout()
        plt.show()
    
    return utility_metrics
