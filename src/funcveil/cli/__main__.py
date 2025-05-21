"""
FuncVeil 명령줄 인터페이스 (CLI)
"""

import click
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any

# 패키지 import를 위한 경로 설정
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from funcveil.masking.sin import SinMask
from funcveil.masking.logexp import LogExpMask
from funcveil.masking.composite import CompositeMask
from funcveil.masking.hmac_mask import HMACMask
from funcveil.evaluation.reversibility import calculate_invertibility_score, visualize_mask_function
from funcveil.evaluation.dp_sensitivity import evaluate_dp_sensitivity, compare_dp_impact
from funcveil.evaluation.benchmarks import compare_masking_methods, generate_benchmark_summary


@click.group()
def cli():
    """FuncVeil: 미적분 기반 비가역 개인정보 마스킹 시스템"""
    pass


@cli.command('mask')
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--column', '-c', multiple=True, help='마스킹할 열 이름 (여러 개 지정 가능)')
@click.option('--method', '-m', type=click.Choice(['sin', 'logexp', 'composite', 'hmac']), 
              default='sin', help='마스킹 방법')
@click.option('--seed', '-s', type=int, default=42, help='난수 생성 시드')
@click.option('--epsilon', '-e', type=float, help='차분 프라이버시 엡실론 값')
@click.option('--monotonic', is_flag=True, help='단조 구간 제한 적용 (sin 마스킹 시)')
@click.option('--hmac-key', type=str, default='default_key', help='HMAC 키 (hmac 마스킹 시)')
@click.option('--invertible', is_flag=True, help='역변환 가능 모드 활성화')
@click.option('--config', '-f', type=click.Path(exists=True), help='JSON 설정 파일')
def mask_command(
    input_file: str, 
    output_file: str, 
    column: List[str], 
    method: str, 
    seed: int,
    epsilon: Optional[float],
    monotonic: bool,
    hmac_key: str,
    invertible: bool,
    config: Optional[str]
):
    """CSV 파일의 지정된 열을 마스킹하여 새 파일로 저장"""
    # 설정 파일 처리
    mask_config = {}
    if config:
        with open(config, 'r') as f:
            mask_config = json.load(f)
    # 데이터 로드
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        click.echo(f"오류: CSV 파일을 읽을 수 없습니다 - {str(e)}", err=True)
        sys.exit(1)
    
    # 마스킹할 열 확인
    columns_to_mask = list(column)
    if not columns_to_mask:
        if 'columns' in mask_config:
            columns_to_mask = mask_config['columns']
        else:
            click.echo("오류: 마스킹할 열을 지정해야 합니다", err=True)
            sys.exit(1)
    
    # 존재하지 않는 열 확인
    missing_cols = [col for col in columns_to_mask if col not in df.columns]
    if missing_cols:
        click.echo(f"오류: 다음 열이 데이터프레임에 존재하지 않습니다: {', '.join(missing_cols)}", err=True)
        sys.exit(1)
    
    # 각 열에 대해 마스킹 적용
    for col in columns_to_mask:
        # 숫자 데이터만 처리 가능
        if not pd.api.types.is_numeric_dtype(df[col]):
            click.echo(f"경고: '{col}' 열은 숫자형이 아니므로 마스킹을 건너뜁니다", err=True)
            continue
        
        # 열별 설정 적용
        col_config = {}
        col_method = method
        col_seed = seed
        col_epsilon = epsilon
        col_monotonic = monotonic
        col_hmac_key = hmac_key
        
        if 'column_config' in mask_config and col in mask_config['column_config']:
            col_config = mask_config['column_config'][col]
            col_method = col_config.get('method', method)
            col_seed = col_config.get('seed', seed)
            col_epsilon = col_config.get('epsilon', epsilon)
            col_monotonic = col_config.get('monotonic', monotonic)
            col_hmac_key = col_config.get('hmac_key', hmac_key)
        # 마스킹 함수 생성
        data = df[col].dropna().values
        
        if col_method == 'sin':
            mask_func = SinMask(
                seed=col_seed, 
                dp_epsilon=col_epsilon, 
                monotonic=col_monotonic
            )
        elif col_method == 'logexp':
            mask_func = LogExpMask(
                seed=col_seed, 
                dp_epsilon=col_epsilon, 
                use_log=invertible
            )
        elif col_method == 'composite':
            mask_func = CompositeMask(
                seed=col_seed, 
                dp_epsilon=col_epsilon
            )
        elif col_method == 'hmac':
            mask_func = HMACMask(
                seed=col_seed, 
                dp_epsilon=col_epsilon, 
                hmac_key=col_hmac_key
            )
        else:
            click.echo(f"오류: 알 수 없는 마스킹 방법 '{col_method}'", err=True)
            sys.exit(1)
        
        # 마스킹 적용
        mask_func.fit(data)
        df[f"{col}_masked"] = df[col].apply(lambda x: mask_func(x) if pd.notnull(x) else None)
        
        click.echo(f"'{col}' 열을 '{col_method}' 방법으로 마스킹했습니다 -> '{col}_masked'")
    
    # 결과 저장
    df.to_csv(output_file, index=False)
    click.echo(f"마스킹된 데이터를 '{output_file}'에 저장했습니다")


@cli.command('eval')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--column', '-c', required=True, help='평가할 원본 열 이름')
@click.option('--masked-column', '-m', help='비교할 마스킹된 열 이름 (지정하지 않으면 column_masked)')
@click.option('--method', type=click.Choice(['sin', 'logexp', 'composite', 'hmac']), 
              default='sin', help='새 마스킹 방법 (마스킹된 열이 없을 경우)')
@click.option('--seed', '-s', type=int, default=42, help='난수 생성 시드')
@click.option('--epsilon-values', '-e', help='차분 프라이버시 엡실론 값 목록 (쉼표로 구분)')
@click.option('--monotonic', is_flag=True, help='단조 구간 제한 적용 (sin 마스킹 시)')
@click.option('--invertible', is_flag=True, help='역변환 가능 모드 활성화')
@click.option('--output-file', '-o', type=click.Path(), help='결과 저장 파일')
def eval_command(
    input_file: str, 
    column: str, 
    masked_column: Optional[str], 
    method: str, 
    seed: int,
    epsilon_values: Optional[str],
    monotonic: bool,
    invertible: bool,
    output_file: Optional[str]
):
    """마스킹 결과 평가 및 분석"""
    # 데이터 로드
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        click.echo(f"오류: CSV 파일을 읽을 수 없습니다 - {str(e)}", err=True)
        sys.exit(1)
    
    # 열 확인
    if column not in df.columns:
        click.echo(f"오류: '{column}' 열이 데이터프레임에 존재하지 않습니다", err=True)
        sys.exit(1)
    
    # 마스킹된 열 확인 또는 생성
    if masked_column is None:
        masked_column = f"{column}_masked"
    
    create_new_mask = False
    if masked_column not in df.columns:
        click.echo(f"'{masked_column}' 열이 존재하지 않습니다. 새로 마스킹을 생성합니다...")
        create_new_mask = True
    
    # 숫자 데이터만 처리 가능
    if not pd.api.types.is_numeric_dtype(df[column]):
        click.echo(f"오류: '{column}' 열은 숫자형이 아닙니다", err=True)
        sys.exit(1)
    
    # 엡실론 값 처리
    eps_list = [0.1, 0.5, 1.0, 5.0, 10.0]  # 기본값
    if epsilon_values:
        try:
            eps_list = [float(e.strip()) for e in epsilon_values.split(',')]
        except ValueError:
            click.echo(f"오류: 엡실론 값 목록 형식이 잘못되었습니다: {epsilon_values}", err=True)
            sys.exit(1)
    
    # 마스킹 함수 생성 (필요한 경우)
    data = df[column].dropna().values
    if create_new_mask:
        # 마스킹 함수 생성
        if method == 'sin':
            mask_func = SinMask(
                seed=seed, 
                monotonic=monotonic
            )
        elif method == 'logexp':
            mask_func = LogExpMask(
                seed=seed, 
                use_log=invertible
            )
        elif method == 'composite':
            mask_func = CompositeMask(
                seed=seed
            )
        elif method == 'hmac':
            mask_func = HMACMask(
                seed=seed
            )
        else:
            click.echo(f"오류: 알 수 없는 마스킹 방법 '{method}'", err=True)
            sys.exit(1)
        
        # 마스킹 적용
        mask_func.fit(data)
        df[masked_column] = df[column].apply(lambda x: mask_func(x) if pd.notnull(x) else None)
    
    # 평가 작업 수행
    # 1. 시각화
    click.echo("마스킹 함수 시각화 중...")
    
    # 동일한 마스킹 함수 생성 (시각화용)
    if method == 'sin':
        mask_func = SinMask(seed=seed, monotonic=monotonic)
    elif method == 'logexp':
        mask_func = LogExpMask(seed=seed, use_log=invertible)
    elif method == 'composite':
        mask_func = CompositeMask(seed=seed)
    elif method == 'hmac':
        mask_func = HMACMask(seed=seed)
    
    mask_func.fit(data)
    visualize_mask_function(mask_func)
    
    # 2. 역변환 가능성 점수
    click.echo("역변환 가능성 평가 중...")
    inv_score = calculate_invertibility_score(mask_func, plot=True)
    click.echo(f"역변환 가능성 점수: {inv_score:.2f} (1.0은 일대일 대응, 값이 클수록 역변환 어려움)")
    # 3. DP 민감도 평가
    click.echo("차분 프라이버시 민감도 평가 중...")
    dp_stats = evaluate_dp_sensitivity(mask_func, epsilon_values=eps_list)
    
    # 결과 표시
    click.echo(f"민감도(Δf): {dp_stats['sensitivity']:.4f}")
    click.echo("엡실론별 통계:")
    for eps, stats in dp_stats['epsilon_stats'].items():
        click.echo(f"  ε = {eps}: 스케일(b) = {stats['scale']:.4f}, 평균 노이즈 = {stats['abs_mean']:.4f}")
    
    # 4. DP 영향 비교
    click.echo("차분 프라이버시 영향 시각화 중...")
    compare_dp_impact(mask_func, epsilon_values=eps_list)
    
    # 결과 저장 (선택적)
    if output_file:
        if create_new_mask:
            df.to_csv(output_file, index=False)
            click.echo(f"마스킹된 데이터와 평가 결과를 '{output_file}'에 저장했습니다")


@cli.command('compare')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--column', '-c', required=True, help='비교할 원본 열 이름')
@click.option('--methods', '-m', default='sin,logexp,composite,hmac', 
              help='비교할 마스킹 방법 목록 (쉼표로 구분)')
@click.option('--seed', '-s', type=int, default=42, help='난수 생성 시드')
@click.option('--epsilon', '-e', type=float, help='차분 프라이버시 엡실론 값')
@click.option('--output-file', '-o', type=click.Path(), help='결과 저장 파일')
def compare_command(
    input_file: str, 
    column: str, 
    methods: str, 
    seed: int,
    epsilon: Optional[float],
    output_file: Optional[str]
):
    """여러 마스킹 방법 비교 분석"""
    # 데이터 로드
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        click.echo(f"오류: CSV 파일을 읽을 수 없습니다 - {str(e)}", err=True)
        sys.exit(1)
    # 열 확인
    if column not in df.columns:
        click.echo(f"오류: '{column}' 열이 데이터프레임에 존재하지 않습니다", err=True)
        sys.exit(1)
    
    # 숫자 데이터만 처리 가능
    if not pd.api.types.is_numeric_dtype(df[column]):
        click.echo(f"오류: '{column}' 열은 숫자형이 아닙니다", err=True)
        sys.exit(1)
    
    # 마스킹 방법 파싱
    method_list = [m.strip() for m in methods.split(',')]
    valid_methods = ['sin', 'logexp', 'composite', 'hmac']
    invalid_methods = [m for m in method_list if m not in valid_methods]
    
    if invalid_methods:
        click.echo(f"오류: 알 수 없는 마스킹 방법: {', '.join(invalid_methods)}", err=True)
        sys.exit(1)
    
    # 데이터 가져오기
    data = df[column].dropna().values
    data_range = (min(data), max(data))
    
    # 마스킹 함수 딕셔너리 생성
    masks = {}
    
    for method in method_list:
        if method == 'sin':
            masks['SinMask'] = SinMask(seed=seed, dp_epsilon=epsilon)
        elif method == 'logexp':
            masks['LogExpMask'] = LogExpMask(seed=seed, dp_epsilon=epsilon)
        elif method == 'composite':
            masks['CompositeMask'] = CompositeMask(seed=seed, dp_epsilon=epsilon)
        elif method == 'hmac':
            masks['HMACMask'] = HMACMask(seed=seed, dp_epsilon=epsilon)
    
    # 비교 실행
    click.echo("마스킹 방법 비교 중...")
    benchmark_results = compare_masking_methods(masks, data_range)
    
    # 요약 표 생성
    summary_df = generate_benchmark_summary(benchmark_results)
    click.echo("\n비교 결과 요약:")
    click.echo(summary_df)
    # 결과 저장 (선택적)
    if output_file:
        # 마스킹 결과 저장
        for method, mask in masks.items():
            mask.fit(data)
            df[f"{column}_{method}"] = df[column].apply(lambda x: mask(x) if pd.notnull(x) else None)
        
        df.to_csv(output_file, index=False)
        click.echo(f"마스킹된 데이터를 '{output_file}'에 저장했습니다")


@cli.command('demo')
@click.option('--output-dir', '-o', type=click.Path(), default='./demo_output', 
              help='데모 출력 저장 디렉토리')
@click.option('--seed', '-s', type=int, default=42, help='난수 생성 시드')
def demo_command(output_dir: str, seed: int):
    """전체 마스킹 및 평가 과정을 시연하는 데모"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 샘플 데이터 생성
    click.echo("샘플 데이터 생성 중...")
    
    # 샘플 데이터프레임 생성
    np.random.seed(seed)
    n_samples = 100
    data = {
        'id': list(range(1, n_samples + 1)),
        'age': np.random.randint(20, 70, n_samples),
        'salary': np.random.normal(50000, 15000, n_samples),
        'score': np.random.uniform(0, 100, n_samples)
    }
    df = pd.DataFrame(data)
    
    # 샘플 데이터 저장
    sample_file = os.path.join(output_dir, 'sample_data.csv')
    df.to_csv(sample_file, index=False)
    click.echo(f"샘플 데이터를 '{sample_file}'에 저장했습니다")
    
    # 2. 다양한 마스킹 방법 적용
    click.echo("\n다양한 마스킹 방법 적용 중...")
    
    # 각 열에 대해 다른 마스킹 방법 적용
    # age: sin 마스킹
    age_mask = SinMask(seed=seed)
    age_mask.fit(df['age'].values)
    df['age_masked'] = df['age'].apply(lambda x: age_mask(x))
    
    # salary: logexp 마스킹
    salary_mask = LogExpMask(seed=seed)
    salary_mask.fit(df['salary'].values)
    df['salary_masked'] = df['salary'].apply(lambda x: salary_mask(x))
    
    # score: composite 마스킹
    score_mask = CompositeMask(seed=seed)
    score_mask.fit(df['score'].values)
    df['score_masked'] = df['score'].apply(lambda x: score_mask(x))
    
    # id: hmac 마스킹
    id_mask = HMACMask(seed=seed, hmac_key='demo_secret_key')
    id_mask.fit(df['id'].values)
    df['id_masked'] = df['id'].apply(lambda x: id_mask(x))
    # 마스킹된 데이터 저장
    masked_file = os.path.join(output_dir, 'masked_data.csv')
    df.to_csv(masked_file, index=False)
    click.echo(f"마스킹된 데이터를 '{masked_file}'에 저장했습니다")
    
    # 3. 마스킹 방법 비교
    click.echo("\n마스킹 방법 비교 중...")
    
    # 마스킹 함수 딕셔너리
    masks = {
        'SinMask': age_mask,
        'LogExpMask': salary_mask,
        'CompositeMask': score_mask,
        'HMACMask': id_mask
    }
    
    # 각 마스킹 방법에 대해 평가 실행
    for name, mask in masks.items():
        click.echo(f"\n{name} 평가:")
        
        # 데이터 범위
        data_col = name.lower().replace('mask', '')
        if data_col == 'hmac':
            data_col = 'id'
        elif data_col == 'logexp':
            data_col = 'salary'
        elif data_col == 'composite':
            data_col = 'score'
        
        data = df[data_col].values
        
        # 역변환 가능성 점수
        inv_score = calculate_invertibility_score(mask, plot=False)
        click.echo(f"  - 역변환 가능성 점수: {inv_score:.2f}")
        
        # 민감도
        sensitivity = mask.get_sensitivity()
        click.echo(f"  - 민감도(Δf): {sensitivity:.4f}")
    
    # 4. 차분 프라이버시 효과 시연
    click.echo("\n차분 프라이버시 효과 시연 중...")
    
    # 다양한 엡실론 값으로 마스킹 생성
    eps_values = [0.1, 1.0, 10.0]
    
    for eps in eps_values:
        # age에 대해 DP 적용
        dp_mask = SinMask(seed=seed, dp_epsilon=eps)
        dp_mask.fit(df['age'].values)
        df[f'age_dp_{eps}'] = df['age'].apply(lambda x: dp_mask(x))
    # DP 적용 데이터 저장
    dp_file = os.path.join(output_dir, 'dp_data.csv')
    df.to_csv(dp_file, index=False)
    click.echo(f"차분 프라이버시 적용 데이터를 '{dp_file}'에 저장했습니다")
    
    # 5. DP 시각화
    click.echo("\nDP 마스킹 평가 및 시각화 중...")
    evaluate_dp_sensitivity(dp_mask, epsilon_values=eps_values)
    compare_dp_impact(dp_mask, epsilon_values=eps_values)
    
    click.echo("\n데모가 완료되었습니다! 결과는 다음 위치에서 확인할 수 있습니다:")
    click.echo(f"  - 샘플 데이터: {sample_file}")
    click.echo(f"  - 마스킹된 데이터: {masked_file}")
    click.echo(f"  - DP 적용 데이터: {dp_file}")


if __name__ == '__main__':
    cli()
