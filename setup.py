from setuptools import setup, find_packages

setup(
    name='credit-risk-scoring',
    version='1.0.0',
    description='Production-ready credit risk scoring model',
    author='Uray Öztürk',
    author_email='urayozturk@icloud.com',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.2.0',
        'scipy>=1.10.0',
    ],
)