from setuptools import setup, find_packages

setup(
    name='needlehaystack',
    version='0.1.0',
    author='Greg Kamradt',
    author_email='greg@gregkamradt.com',
    description='Doing simple retrieval from LLM models at various context lengths to measure accuracy.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gkamradt/LLMTest_NeedleInAHaystack',
    packages=find_packages(),
    include_package_data=True,  # This includes non-code files specified in MANIFEST.in
    install_requires=[
        x for x in open("./requirements.txt", "r+").readlines() if x.strip()
    ],
    python_requires='>=3.6',
    classifiers=[],
    entry_points={
        'console_scripts': [
            'needlehaystack.run_test = needlehaystack.run:main',
        ],
    },
)
