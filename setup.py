from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="luxembourgish-vowel-classifier",
    version="0.1.0",
    author="Peter Gilles",
    author_email="peter.gilles@uni.lu",
    description="A Streamlit app for classifying Luxembourgish vowels using HuBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PeterGilles/luxembourgish-vowel-classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vowel-classifier=app.streamlit_app:main",
        ],
    },
)