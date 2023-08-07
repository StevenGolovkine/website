---
title: Set up Python
author: ~
date: '2018-12-06'

slug: set-up-python
categories: ['Python']
tags: ['Python', 'Set up']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true

Summary: Python is a programming language. It is used by a lot of people doing data science.
---


[Python](https://www.python.org) is already installed on macOS. But as we do not want to mess with it, and have more flexibility, we will install our own version of **Python**. This post is based on this [post](https://medium.com/@henriquebastos/the-definitive-guide-to-setup-my-python-workspace-628d68552e14) by Henrique Bastos on Medium.

## Wanted configurations

* **CPython 2.7** and **CPython 3.7**, but it is possible to install other implementations like **PyPy** or **Anaconda**.
* **Python3** as a default version for everything, but it must easily change to **Python2**.
* A *unique* **Jupyter Notebook/Lab** working with both **Python2** and **Python3**, and being able to detect the active virtual environment.
* A console *iPython* for **Python3** and one for **Python2**, so no need to install it in every virtual environment of the projects.
* *virtualenvwrapper* to develop the different projects and change the context in one command.

## Installation

*pyenv* is probably the best way to install **Python** on macOS. Everything should be installed in the common directory without interfering with the rest of the system. Moreover, it handles with a lot of **Python** implementation such as **CPython**, **PyPy**, **Anaconda**, etc. And all of that with only one command. Firstly, one should install *pyenv* and two add-ons:

* _pyenv_ to install **Python** implementations;
* _pyenv-virtualenv_ to configure global environment;
* _pyenv-virtualenvwrapper_ to work on projects.

```bash
brew install pyenv
brew install pyenv-virtualenv
brew install pyenv-virtualenvwrapper
```

With _virtualenvwrapper_, every _virtualenv_ will be kept in the same repository and every projects codes in an other.

```bash
# Every virtual environment will be in ...
mkdir ~/.ve
# Every projects will be in ...
mkdir ~/Documents/Python/workspace
```

We have to configure the _shell_ to initialise _pyenv_ at the opening of the terminal. Thus, we have to add the following lines into the file `~/.bashrc`.

```bash
export WORKON_HOME=~/.ve
export PROJECT_HOME=~/Documents/Python/workspace
eval "$(pyenv init -)"
```

Reload the terminal to take the changes into account.

Next step is to install **CPython 3.7.1** and **CPython 2.7.15**.

```bash
pyenv install 3.7.1
pyenv install 2.7.15
```

## Configure the global Python installation

It is nice to use **Python** written programs without using a virtual environment. Moreover, it is easier if we only have one _Jupyter Notebook/Lab_, one _iPython console_ for **Python 2**, one _iPython console_ for **Python 3** and other tools.

So, we use _pyenv-virtualenv_ to do that:
```bash
pyenv virtualenv 3.7.1 jupyter3
pyenv virtualenv 3.7.1 tools3
pyenv virtualenv 2.7.15 ipython2
pyenv virtualenv 2.7.15 tools2
```

_Jupyter_ can handle with many kernels like **Python 2**, **Python 3**, **R**, **bash**, and some other. It allows only one _Jupyter_ installation.

> Here, we just want to use **Python2** and **Python3**.

Let's start with **Python3**:
```bash
pyenv activate jupyter3
pip install jupyter
pip install jupyterlab
python -m ipykernel install --user
pyenv deactivate
```

Let's continue with **Python2**:
```bash
pyenv activate ipython2
pip install ipykernel
python -m ipykernel install --user
pyenv deactivate
```

> Note that when we install _Jupyter_ for **Python3**, we install by default _iPython_ and the _kernel_. For **Python2**, we only need to install _iPython_ and the _kernel_.

Now, let's install tools using **Python3**:
```bash
pyenv activate tools3
pip install youtube-dl rows
pyenv deactivate
```

Let's install tools which do not work with **Python3** but only with **Python2**:
```bash
pyenv activate tools2
pip install rename
pyenv deactivate
```

Finally, it is time to let all the **Python** versions and the special virtual environments working together.

```bash
pyenv global 3.7.1 2.7.15 jupyter3 ipython2 tools3 tools2
```

> Note that this command put priority in the `$PATH` environment variable. Thus, it is possible to reach the scripts without activating virtual environments.


## Using virtual environment

We use _pyenv-virtualenvwrapper_ to create the virtual environment for each project. Now, he have to add the line `pyenv virtualenvwrapper_lazy` in the file `~/.bashrc` and then reload the terminal. When we start a new session, _pyenv-virtualenvwrapper_ will install the necessary dependencies of _virtualenvwrapper_ if they are not here. It is possible to use commands from _virtualenvwrapper_ and every virtual environment will be created by using **Python** implementations installed from _pyenv_.

Some examples:

1. Let's say I want a new project _proj3_ using **Python3**. The command `mkproject proj3` will create a new virtual environment using **Python3** (by default) in the repository `~/.ve/proj3` and a project repository `~/Documents/Python/workspace/proj3`.

2. Let's imagine I want to work on my project _proj3_. Run the command `workon proj3` will activate the virtual environment `~/.ve/proj3` and change the working directory to `~/Documents/Python/workspace/proj3`.

3. Let's clone a project names _proj2_ in the directory `~/Documents/Python/workspace/proj2`. So, I need a virtual environment for this project. Run the command `mkvirtualenv -a ~/Documents/Python/workspace/proj2 -p python2 proj2` will create a virtual environment using **Python2** in the directory `~/.ve/proj2` linked to the project. Then, run `workon proj2` will activate the virtual environment and change the working directory.

## Using Jupyter and iPython with the projects

> At the beginning, _Jupyter_ and _Console_ were parts of _iPython_ project which was only about **Python**. But the evolution of the _Notebook_ allows to use more languages than just **Python**. So, the developers decide to split the project: _Jupyter_ and _iPython_. Now, _Notebook_ is part of _Jupyter_ and _Console_ is part of _iPython_ and the _Python kernel_ used by _Jupyter_ to launch **Python** code.

So, _Jupyter_ do not detect the active virtual environment: it is the _iPython_ instance that _Jupyter_ initialize. The problem is that the _iPython_ virtual environment launches itself only in _interactive shell_ mode and not in _kernel_ mode. Otherwise, the code detection works correctly only if the **Python** version of the active virtual environment and the **Python** version launches by _iPython_ are the same.

The solution is to customize the process of the _iPython_ start-up. To do that, we need a _iPython profile_ et launch a [script](https://gist.github.com/henriquebastos/270cff100cb303f3d74370489022446b):
```bash
ipython profile create
curl -L http://hbn.link/hb-ipython-startup-script > ~/.ipython/profile_default/startup/00-venv-sitepackages.py
```
So, no matter the mode in which _iPython_ is launched, the _site-packages_ of the virtual environment will be available in the `PYTHONPATH`.

Back to _proj3_, after run `workon proj3`, it is possible to execute `iPython` to be in the interactive mode, or `jupyter-notebook` to use the notebook.

## Updating the package with pip

In order to update the different **Python** packages, run the following command:
```bash
pip list --outdated | cut -d " " -f 1 | xargs -n1 pip install -U
```
