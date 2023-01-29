from setuptools import setup, find_packages

REQUIRED_PKGS = [
    "transformers==4.26.0",
    "accelerate==0.13.2",
    "datasets==2.6.1",
    "evaluate==0.3.0",
    "pyext==0.7",
    "mosestokenizer==1.0.0",
    "huggingface_hub==0.11.1",
]

DS100_REQUIRE = [
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
    "tqdm==4.64.1",
    "xgboost==1.6.2",
    "Pillow==9.2.0",
]

EXTRAS_REQUIRE = {"ds1000": DS100_REQUIRE}

setup(
    name="bigcode-eval",
    version="0.1",
    description="Framework for the evaluation of code generation models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Loubna Ben Allal, Niklas Muennighoff",
    author_email="loubna@huggingface.co, niklas@huggingface.co",
    url="https://github.com/bigcode-project/bigcode-evaluation-harness",
    license="Apache 2.0",
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.7.0",
    package_dir={"": "lm_eval"},
    packages=find_packages("lm_eval"),    
)
