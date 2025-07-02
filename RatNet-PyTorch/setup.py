from setuptools import setup, find_packages

setup(
    name="radnet-pytorch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "nuscenes-devkit>=1.1.9",
        "numpy>=1.19.2",
        "Pillow>=8.0.0",
        "PyYAML>=5.4.1",
        "tensorboard>=2.4.0",
        "tqdm>=4.50.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="PyTorch implementation of RadNet++ for radar-camera calibration",
    keywords="deep-learning, calibration, radar, camera, pytorch",
    python_requires=">=3.7",
) 