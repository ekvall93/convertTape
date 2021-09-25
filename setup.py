from setuptools import setup

setup(
    name="torchToTF_tape",
    version="1.0",
    py_modules=['convert'],
    install_requires=[
        'Click',
        'tape_proteins @ https://github.com/ekvall93/tape/tarball/master',
        'tensorflow-gpu==2.4.1',
        'torch==1.8.1',
        'tqdm==4.62.3',
        'onnx==1.9.0',
        'onnx-tf==1.8.0',
        'scikit-learn==0.24.2',
        'scipy==1.5.4'
    ],
    entry_points='''
    [console_scripts]
    convert=convert:cli
    validate=validate:cli
    '''
)
