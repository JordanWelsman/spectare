"""
Setup file for the package.
"""


# Module Imports
from setuptools import setup, find_packages


# Define list of submodules
py_modules = ["spectare"]

# Module Setup
setup(
    name="spectare",
    version="0.0.0",
    author="Jordan Welsman",
    author_email="jordan.welsman@outlook.com",
    description="A PyTorch visualisation and interpretability framework.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JordanWelsman/spectare",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Utilities"
    ],
    package_data = {
        'spectare': py_modules
        },
    python_requires='>=3.10',
    install_requires=[
        "networkx>=3.2.1"
    ]
)
