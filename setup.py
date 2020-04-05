import os
import re
import ast
import setuptools

_this_dir = os.path.abspath(os.path.dirname(__file__))

# parse version from ./laser/__init__.py
_version_re = re.compile(r'__version__\s+=\s+(.*)')
_init_file = os.path.join(_this_dir, 'laser', '__init__.py')
with open(_init_file, 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))


# parse requirements form ./requirements.txt
_requirements_file = os.path.join(_this_dir, 'requirements.txt')
with open(_requirements_file, 'rb') as f:
    install_requires = [l.decode('utf-8').strip() for l in f]


setuptools.setup(
    # General information
    name='laser-keep-alive',
    version=version,
    description='Keeping the original LASER project alive',
    author='Wang M. R.',
    author_email='mingruimingrui@hotmail.com',
    url='https://github.com/mingruimingrui/laser-keep-alive',

    # Build arguments
    packages=setuptools.find_packages(exclude=['models', 'tests']),
    python_requires='>=3.6.1',
    install_requires=install_requires,

    # PyPI package information
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='laser sentence embedding',
    license='MIT',
)