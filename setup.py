from setuptools import setup, find_packages

setup(
    name="pyrise",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "jax[cpu]>=0.4.0",
        "dynamax>=0.1.0",
        "pandas>=2.0.0",
        "ruamel-yaml>=0.17.0",
        "lark>=1.1.0",
        "sympy>=1.12.0",
        "scipy>=1.11.0",
        # add any other deps here
    ],
    entry_points={
        "console_scripts": [
            "pyrise=pyrise.cli:main",
        ],
    },
)
