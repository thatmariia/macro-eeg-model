from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return lines


setup(
    name="macro-eeg-model",
    version="0.1",
    description="Modeling EEG with axon propagation delays",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Mariia Steeghs-Turchina",
    author_email="m.turchina@uva.nl",
    url="https://github.com/thatmariia/macro-eeg-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        'console_scripts': [
            'py_simulate = simulate:simulate',
            'py_evaluate = evaluate:evaluate',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10"
        "License :: MIT License",
    ],
    python_requires=">=3.10"
)