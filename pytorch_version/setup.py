import setuptools

__version__ = "0.1.0"

setuptools.setup(
    name="rocf_scoring",
    version=__version__,
    python_requires=">=3.6",
    packages=setuptools.find_packages(
        exclude=["data", "scripts"]
    ),
)
