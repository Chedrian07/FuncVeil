"""
마스킹 기법 벤치마크 및 비교 도구
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any

from funcveil.masking.base import IFitMask
from funcveil.evaluation.reversibility import calculate_invertibility_score


def compare_masking_methods(
    masks: Dict[str, IFitMask],
    data_range: Tuple[float, float],
    samples: int = 100,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """여러 마스킹 방법 비교 및 시각화

    Args:
        masks: 마스킹 객체 딕셔너리 (이름: 객체)
        data_range: 데이터 범위 (min, max)
        samples: 테스트할 샘플 수
        figsize: 그래프 크기
        save_path: 그래프 저장 경로 (None이면 저장 안함)

    Returns:
        벤치마크 결과 딕셔너리
    """
    # 테스트 데이터 생성
    x_vals = np.linspace(data_range[0], data_range[1], samples)
    
    # 결과 저장 딕셔너리
    results = {}
    
    # 각 마스킹 방법 초기화 및 적용
    for name, mask in masks.items():
        # 필요시 초기화
        if not mask.initialized:
            mask.fit(x_vals)
        
        # 시간 측정
        start_time = time.time()
        y_vals = [float(str(mask(x)).split(':')[-1]) if ':' in str(mask(x)) 
                 else float(mask(x)) for x in x_vals]
        elapsed_time = time.time() - start_time
        
        # 민감도 계산
        sensitivity = mask.get_sensitivity()
        
        # 역변환 가능성 점수 계산
        inv_result = calculate_invertibility_score(mask, samples=samples, plot=False, return_dict=True)
        inv_score = inv_result['overall_score']
        
        # 결과 저장
        results[name] = {
            'x_vals': x_vals,
            'y_vals': y_vals,
            'elapsed_time': elapsed_time,
            'sensitivity': sensitivity,
            'invertibility_score': inv_score,
            'invertibility_details': inv_result,  # 상세 정보 저장
            'range': (min(y_vals), max(y_vals)),
            'mean': np.mean(y_vals),
            'std': np.std(y_vals)
        }
    
    # 결과 시각화
    plt.figure(figsize=figsize)
    
    # 마스킹 함수 시각화
    plt.subplot(2, 2, 1)
    for name, res in results.items():
        plt.plot(res['x_vals'], res['y_vals'], '-', linewidth=2, label=name)
    
    plt.title('Masking Functions Comparison')
    plt.xlabel('Input Value')
    plt.ylabel('Masked Value')
    plt.legend()
    plt.grid(True)
    
    # 민감도 비교
    plt.subplot(2, 2, 2)
    names = list(results.keys())
    sensitivities = [results[name]['sensitivity'] for name in names]
    
    plt.bar(names, sensitivities)
    plt.title('Sensitivity Comparison')
    plt.xlabel('Masking Method')
    plt.ylabel('Sensitivity (Max Derivative)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 역변환 가능성 비교
    plt.subplot(2, 2, 3)
    inv_scores = [results[name]['invertibility_score'] for name in names]
    
    plt.bar(names, inv_scores)
    plt.title('Invertibility Score Comparison')
    plt.xlabel('Masking Method')
    plt.ylabel('Invertibility Score')
    plt.xticks(rotation=45)
    plt.grid(True)
    # 실행 시간 비교
    plt.subplot(2, 2, 4)
    times = [results[name]['elapsed_time'] * 1000 for name in names]  # ms 단위로 변환
    
    plt.bar(names, times)
    plt.title('Execution Time Comparison')
    plt.xlabel('Masking Method')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    
    # 그래프 저장 (선택적)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"벤치마크 결과를 '{save_path}'에 저장했습니다.")
    
    plt.show()
    
    return results


def generate_benchmark_summary(benchmark_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """벤치마크 결과 요약 표 생성

    Args:
        benchmark_results: compare_masking_methods의 반환값

    Returns:
        요약 데이터프레임
    """
    summary_data = []
    
    for name, res in benchmark_results.items():
        summary_data.append({
            'Method': name,
            'Sensitivity': res['sensitivity'],
            'Invertibility Score': res['invertibility_score'],
            'Output Range': f"{res['range'][0]:.2f} - {res['range'][1]:.2f}",
            'Execution Time (ms)': res['elapsed_time'] * 1000,
            'Output Mean': res['mean'],
            'Output Std': res['std']
        })
    
    return pd.DataFrame(summary_data)
