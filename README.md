# FuncVeil: 미적분 기반 비가역 개인정보 마스킹 시스템

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

FuncVeil은 고등학교 수학 교육과정에서 배우는 미적분 개념을 활용하여 개인정보를 안전하게 마스킹하는 Python 라이브러리입니다. 다양한 수학 함수와 차분 프라이버시를 활용하여 통계적 유용성을 유지하면서도 복호화가 불가능한 데이터 변환을 제공합니다.

## 주요 기능

- **연속함수 기반 마스킹**: 사인, 코사인, 지수, 로그 함수 활용
- **복합 함수 마스킹**: 여러 함수를 조합한 고급 마스킹
- **HMAC 결합 마스킹**: 암호학적 해시와 수학 함수 결합
- **차분 프라이버시**: 도함수 기반 민감도 계산 및 DP 노이즈 적용
- **마스킹 평가 도구**: 역변환 가능성 점수, DP 효과 시각화

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/funcveil/funcveil.git
cd funcveil

# 패키지 설치
pip install -e .
```

## 사용 예시

### 기본 마스킹 적용

```python
from funcveil.masking.sin import SinMask
import numpy as np

# 데이터 준비
data = np.array([25, 30, 35, 40, 45])

# 마스킹 객체 생성 및 초기화
mask = SinMask(seed=42)
mask.fit(data)

# 마스킹 적용
masked_values = mask.mask_all(data)
print(masked_values)
```

### 명령줄 도구 사용

```bash
# CSV 파일 마스킹
funcveil mask input.csv output.csv -c age -c salary -m sin

# 마스킹 함수 평가
funcveil eval input.csv -c age -e 0.1,0.5,1.0,5.0

# 여러 마스킹 함수 비교
funcveil compare input.csv -c age -m sin,logexp,composite,hmac

# 데모 실행
funcveil demo -o demo_output
```
## 교육적 활용

FuncVeil은 고등학교 교육과정의 다양한 수학 개념을 실생활 문제에 적용하는 방법을 보여줍니다:

1. **연속함수**: 수학적 함수를 데이터 변환에 활용
2. **도함수**: 함수의 민감도를 계산하고 차분 프라이버시에 적용
3. **합성함수**: 다양한 함수를 조합하여 복잡한 패턴 생성
4. **역함수**: 단조 구간 제한으로 역변환 가능성 조절
5. **미분방정식**: 간단한 미분방정식을 활용한 마스킹 함수 구현

자세한 수학적 설명은 [수학 모델 문서](docs/math_model.md)를 참조하세요.

## 응용 분야 및 시나리오

- **학교/학원 출결 시스템**: 학생 정보를 안전하게 보호하면서도 출결 통계 관리
- **금융기관 데이터 처리**: 민감한 금융 정보를 마스킹하여 분석 가능
- **공공기관 데이터 익명화**: 통계 목적으로 공공 데이터를 안전하게 공개

## 프로젝트 구조

```
FuncVeil/
├── src/                      # 소스 코드
│   ├── funcveil/             # 메인 패키지
│   │   ├── masking/          # 마스킹 알고리즘
│   │   ├── evaluation/       # 평가 도구
│   │   └── cli/              # 명령줄 인터페이스
├── tests/                    # 테스트 코드
├── notebooks/                # 예제 노트북
└── docs/                     # 문서
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 참고 자료

- [차분 프라이버시 소개](https://en.wikipedia.org/wiki/Differential_privacy)
- [HMAC 알고리즘](https://en.wikipedia.org/wiki/HMAC)
- [고등학교 수학과 교육과정](https://www.moe.go.kr)
# FuncVeil
