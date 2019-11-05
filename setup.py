import setuptools

setuptools.setup(
    name='torch_intermediate_layer_getter',
    author='Sebastian Amenabar',
    author_email='amenabars@gmail.com',
    description='Simple easy to use module to get the intermediate results from chosen submodules',
    version='0.1.post1',
    packages=['torch_intermediate_layer_getter'],
    url='https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ]
)