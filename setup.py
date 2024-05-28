import setuptools

with open("requirements.txt") as f:
    requires = [x.strip() for x in f.readlines()]

with open("README.md") as file:
    readme = file.read()

with open("HISTORY.md") as file:
    history = file.read()

setuptools.setup(
    name="Lenzer",
    version="0.0.0",
    description="Strong Gravitational Lens Simulation Method",
    url="https://github.com/RubyPC/Lenzer",
    python_requires=">=3.6",
    setup_requires=["pytest-runner"],
    install_requires=requires,
    tests_require=["pytest"],
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    zip_safe=False,
)
