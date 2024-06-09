from setuptools import find_packages, setup

setup(
    name='DISSERTATION_CAMELYON',  # 将 'my_project' 替换为你的项目名称
    version='0.1.0',  # 项目的版本
    packages=find_packages(where='src'),  # 自动查找 'src' 目录中的所有包
    package_dir={'': 'src'},  # 指定包目录
    install_requires=[  # 项目的依赖项
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        # 其他依赖项
    ],
)
