"""
FuncVeil 패키지 설치 스크립트
"""

from setuptools import setup, find_packages

setup(
    name="funcveil",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "funcveil=funcveil.cli.__main__:cli",
        ],
    },
    author="FuncVeil Team",
    author_email="funcveil@example.com",
    description="미적분 기반 비가역 개인정보 마스킹 시스템",
    long_description=open("docs/math_model.md").read(),
    long_description_content_type="text/markdown",
    keywords="privacy, masking, differential-privacy, mathematical-functions",
    url="https://github.com/funcveil/funcveil",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
)
