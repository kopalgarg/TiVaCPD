import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TiVaCPD",
    version="1.0.0",
    author="Kopal Garg",
    author_email="gargkopal24@gmail.com",
    description="Time-Varying Change Point Detection",
    url="https://github.com/kopalgarg/TiVaCPD",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)