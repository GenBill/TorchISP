from setuptools import setup, find_packages

setup(
    name='torchisp',
    version='0.1.1',
    description='A PyTorch-based image signal processing tool for converting RGGB to RGB images.',
    author='GenBill',
    author_email='genbill97@gmail.com',
    url='https://github.com/GenBill/TorchISP',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)
