---
title: Homebrew
author: ~
date: '2018-12-06'

slug: homebrew
categories: ['Software']
tags: ['Software', 'Set up']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true

Summary: Hombrew is a package manager for macOS. This software is a must to have for macOS users.
---

[Homebrew](https://brew.sh) names itself as _the missing package manager for macOS_. It simplifies the installation and the management of the different softwares you could have.

## Installation

First, you need to have the **Command Line Tools** for **Xcode**. The installation of **Xcode** is made from the App Store. Once it is done, you can install the **Command Line Tools** using the following command in the terminal:

```bash
xcode-select --install
```

Then, you should launch the following command to have **Homebrew** installed:

```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

After, you should tell to the system to take into consideration programs installed by **Homebrew** rather than the system default. By default, **Homebrew** uses the `/usr/local/bin` path. We do this by the adding this path to the `$PATH` environment variable.

```bash
echo 'export PATH="/usr/local/bin/:$PATH"' >> ~/.bash_profile
```

## Usage

We install and uninstall a formula by using the commands:

```bash
brew install <formula>
brew uninstall <formula>
```

To upgrade all the formulae, run:

```bash
brew update
brew upgrade
```

To list all the formulae you have with their version, run:

```bash
brew list --versions
```

Finally, **Homebrew** keeps a trace of the previous versions of each of the formula (if you want to get it back). If you want to delete it, run:

```bash
brew cleanup
```

## Cask

[Homebrew-Cask](https://caskroom.github.io) extends **Homebrew** and allows to install software using command-line tools.

To look for a software, run:

```bash
brew cask search <formula>
```

To install/uninstall a software, run:

```bash
brew cask install <formula>
brew cask uninstall <formula>
```

To know the outdated formulae, run:

```bash
brew cask outdated
```

And then, for update the package, run:

```bash
brew cask reinstall <formula>
```
