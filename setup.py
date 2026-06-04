from setuptools import setup, find_packages

setup(
    name="eeg_toolkit",
    version="0.1.0",
    description="EEG preprocessing and analysis toolkit",
    author="Juan Pablo",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "mne>=1.5",
        "numpy",
        "pandas",
        "pyyaml",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "pyxdf",
        "PyQt5",
        "mne-icalabel",
        "rsatoolbox",
    ],
    extras_require={
        "dev": ["pytest", "jupyter"],
    },
)