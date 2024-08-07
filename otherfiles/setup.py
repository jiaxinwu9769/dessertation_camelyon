from setuptools import find_packages, setup

setup(
    name='DISSERTATION_CAMELYON', 
    version='0.1.0',  
    packages=find_packages(where='src'),  
    package_dir={'': 'src'}, 
    install_requires=[  
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        # 
    ],
)
