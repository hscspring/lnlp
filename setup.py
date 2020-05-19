import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lnlp",
    version="0.0",
    author="Yam",
    author_email="haoshaochun@gmail.com",
    description="Last Mile for NLP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hscspring/lnlp",
    packages=setuptools.find_packages(),
    install_requires=[
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)