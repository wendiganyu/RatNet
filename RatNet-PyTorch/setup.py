from setuptools import setup, find_packages

setup(
    name="ratnet-pytorch",
    version="0.1.0",
    description="PyTorch implementation of RadNet++: Geometric supervision model for rotational radar-camera calibration",
    author="PyTorch implementation based on original work by Odysseas Papanikolaou",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "tqdm>=4.50.0",
        "pyyaml>=5.4.0",
        "nuscenes-devkit>=1.1.0",
        "pyquaternion>=0.9.5",
        "tensorboard>=2.4.0",
        "einops>=0.3.0",
        "scikit-learn>=0.24.0"
    ],
) 