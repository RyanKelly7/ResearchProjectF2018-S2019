from setuptools import setup


setup(
    name = "cloudfinder",
    version = "0.0.1",
    author = "Ryan Kelly",
    author_email = "rjk228@nau.edu",
    description = ("Software to detect clouds using allsky cameras"),
    license = "MIT",
    keywords = "Cloud Allsky Camera Detection Finder",
    url = "",
    packages=['cloudfinder'],
    install_requires = ['numpy','astropy','sep'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: MIT",
    ],
)

