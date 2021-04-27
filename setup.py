import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="ranger21",
    version="0.0.1",
    author="lessw2020",
    description="Integrating the latest deep learning components into a single optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/lessw2020/Ranger21",
    project_urls={
        "Bug Tracker": f"https://github.com/lessw2020/Ranger21/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=["ranger21"],
    python_requires=">=3.6",
)
