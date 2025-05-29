from setuptools import setup, find_packages

setup(
    name="multilevel_segmentation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
    ],
    python_requires=">=3.8",
    author="Kingo Kingsley K",
    author_email="kingokingsleykaah.com",
    description="A library for multilevel image segmentation using combined clustering and morphology",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dmkvoltage/multilevel_segmentation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
