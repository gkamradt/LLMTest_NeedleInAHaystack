from setuptools import setup, find_packages

setup(
    name='niah-llm-test',
    version='0.0.1',
    author='Greg Kamradt',
    author_email='greg@gregkamradt.com',
    description='Doing simple retrieval from LLM models at various context lengths to measure accuracy.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gkamradt/LLMTest_NeedleInAHaystack',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    python_requires='>=3.6',
    classifiers=[],
)
