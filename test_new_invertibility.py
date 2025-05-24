#!/usr/bin/env python3
"""
FuncVeil 가역성 평가 테스트 스크립트
새로운 가역성 평가 로직이 올바르게 작동하는지 검증합니다.
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from funcveil.masking.sin import SinMask
from funcveil.masking.logexp import LogExpMask
from funcveil.masking.composite import CompositeMask
from funcveil.masking.hmac_mask import HMACMask
from funcveil.evaluation.reversibility import calculate_invertibility_score
from funcveil.evaluation.benchmarks import compare_masking_methods, generate_benchmark_summary


def test_individual_masks():
    """각 마스킹 함수의 가역성을 개별적으로 테스트"""
    print("=" * 80)
    print("가역성 평가 결과 (새로운 로직)")
    print("=" * 80)
    
    # 데이터 범위 설정
    data_range = (0, 100)
    test_data = np.linspace(data_range[0], data_range[1], 100)
    
    # 마스킹 함수들 생성
    masks = {
        'SinMask': SinMask(bounds=data_range, dp_epsilon=1.0),
        'SinMask_Monotonic': SinMask(bounds=data_range, dp_epsilon=1.0, monotonic=True),
        'LogExpMask': LogExpMask(bounds=data_range, dp_epsilon=1.0),
        'CompositeMask': CompositeMask(bounds=data_range, dp_epsilon=1.0),
        # 'HMACMask': HMACMask(bounds=data_range, dp_epsilon=1.0)  # 일시적으로 제외
    }
    
    # 각 마스크 초기화
    for name, mask in masks.items():
        mask.fit(test_data)
    
    # 각 마스킹 함수에 대해 가역성 평가
    all_scores = {}
    for name, mask in masks.items():
        results = calculate_invertibility_score(mask, samples=200, plot=False, return_dict=True)
        all_scores[name] = results['overall_score']
        
        print(f"\n{name}:")
        print(f"  - 종합 점수: {results['overall_score']:.2f} / 10.0")
        print(f"  - 충돌 계수: {results['collision_factor']:.3f} (1에 가까울수록 좋음)")
        print(f"  - 정보 보존율: {results['information_preservation_rate']:.3f} (1에 가까울수록 좋음)")
        print(f"  - 역변환 정확도: {results['inverse_accuracy']:.3f} (1에 가까울수록 좋음)")
    
    # 점수 다양성 검증
    print("\n" + "=" * 60)
    print("가역성 점수 다양성 검증:")
    print("=" * 60)
    
    score_values = list(all_scores.values())
    score_std = np.std(score_values)
    
    print(f"점수 표준편차: {score_std:.3f}")
    print(f"점수 범위: {min(score_values):.2f} ~ {max(score_values):.2f}")
    
    if score_std < 0.1:
        print("\n⚠️  경고: 가역성 점수가 여전히 너무 유사합니다!")
    else:
        print("\n✅ 성공: 가역성 점수가 함수별로 다르게 계산됩니다!")
    
    return masks, all_scores


def test_detailed_analysis():
    """더 상세한 가역성 분석"""
    print("\n" + "=" * 80)
    print("상세 가역성 분석 (시각화 포함)")
    print("=" * 80)
    
    data_range = (0, 100)
    test_data = np.linspace(data_range[0], data_range[1], 100)
    
    # 대표적인 3개 함수만 상세 분석
    test_masks = {
        'SinMask': SinMask(bounds=data_range, dp_epsilon=1.0),
        'SinMask_Monotonic': SinMask(bounds=data_range, dp_epsilon=1.0, monotonic=True),
        'CompositeMask': CompositeMask(bounds=data_range, dp_epsilon=1.0)
    }
    
    for name, mask in test_masks.items():
        mask.fit(test_data)
        print(f"\n{name} 상세 분석:")
        results = calculate_invertibility_score(mask, samples=200, plot=True, return_dict=True)
        input("다음 그래프를 보려면 Enter를 누르세요...")


def test_benchmark_comparison():
    """종합 벤치마크 테스트"""
    print("\n" + "=" * 80)
    print("종합 벤치마크 비교")
    print("=" * 80)
    
    data_range = (0, 100)
    test_data = np.linspace(data_range[0], data_range[1], 100)
    
    masks = {
        'SinMask': SinMask(bounds=data_range, dp_epsilon=1.0),
        'SinMask_Monotonic': SinMask(bounds=data_range, dp_epsilon=1.0, monotonic=True),
        'LogExpMask': LogExpMask(bounds=data_range, dp_epsilon=1.0),
        'CompositeMask': CompositeMask(bounds=data_range, dp_epsilon=1.0),
        # 'HMACMask': HMACMask(bounds=data_range, dp_epsilon=1.0)  # 일시적으로 제외
    }
    
    # 각 마스크 초기화
    for name, mask in masks.items():
        mask.fit(test_data)
    
    # 벤치마크 실행
    benchmark_results = compare_masking_methods(masks, data_range, samples=200)
    
    # 요약 표 생성
    summary_df = generate_benchmark_summary(benchmark_results)
    print("\n벤치마크 요약:")
    print(summary_df.to_string(index=False))


def main():
    """메인 테스트 함수"""
    print("FuncVeil 가역성 평가 테스트 시작\n")
    
    # 1. 개별 마스크 테스트
    masks, scores = test_individual_masks()
    
    # 2. 상세 분석 (선택적)
    response = input("\n상세 분석을 실행하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        test_detailed_analysis()
    
    # 3. 벤치마크 비교
    response = input("\n종합 벤치마크를 실행하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        test_benchmark_comparison()
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    main() 