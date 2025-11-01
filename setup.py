from setuptools import setup, find_packages

setup(
    name="customer-churn-mlops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    author="Shihasz",
    description="MLOps pipeline for customer churn prediction",
    python_requires=">=3.11",
)