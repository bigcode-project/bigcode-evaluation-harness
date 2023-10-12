from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")

ds1000_requirements = [
    "DateTime==4.7",
    "gensim==4.2.0",
    "matplotlib==3.5.2",
    "numpy==1.21.6",
    "openai==0.23.0",
    "pandas==1.3.5",
    "pandas-datareader==0.10.0",
    "pathlib==1.0.1",
    "scikit-learn==1.0.2",
    "scipy==1.7.3",
    "seaborn==0.11.2",
    "statsmodels==0.13.2",
    "tensorflow==2.10.0",
    "tokenizers==0.12.1",
    "torchvision==0.13.1",
    "tqdm==4.64.1",
    "xgboost==1.6.2",
    "Pillow==9.2.0",
]

setup(
    description="A framework for the evaluation of autoregressive code generation language models.",
    long_description=readme,
    license="Apache 2.0",
    packages=find_packages() ,
    install_requires=requirements,
    extras_require={"ds1000": ds1000_requirements},
)
