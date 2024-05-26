from setuptools import setup, find_packages

setup(
    name="PruneGPT",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers"
    ],
    entry_points={
        "console_scripts": [
            "prune-model=main:main",
        ],
    },
)