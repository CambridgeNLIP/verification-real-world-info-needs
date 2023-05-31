from setuptools import setup, find_packages
import sys


with open('requirements.txt') as f:
    reqs = f.read()

reqs = reqs.strip().split('\n')

install = [req for req in reqs if not req.startswith("git+git://")]
depends = [req.replace("git+git://","git+http://") for req in reqs if req.startswith("git+git://")]

setup(
    name='nc-annotations',
    version='0.0.0',
    author='James Thorne',
    author_email='james@jamesthorne.co.uk',
    url='https://jamesthorne.co.uk',
    description='Fact Extraction and VERification annotation',
    long_description="readme",
    license=license,
    package_dir={},
    packages=[],
    install_requires=install,
    dependency_links=depends,
)