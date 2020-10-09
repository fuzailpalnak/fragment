from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
install_requires = [
    "numpy == 1.19.1",
]

setup(
    name="image_fragment",
    version="0.2.2",
    author="Fuzail Palnak",
    author_email="fuzailpalnak@gmail.com",
    url="https://github.com/fuzailpalnak/fragment",
    description="Data Section",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires="~=3.6",
    install_requires=install_requires,
    keywords=["numpy, Window, Section, Array, Stitch, Split"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
