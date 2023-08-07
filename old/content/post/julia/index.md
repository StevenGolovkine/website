---
title: Set up Julia
author: ~
date: '2019-12-30'

slug: set-up-julia
categories: ['Julia', 'Programming Language', 'Set up']
tags: ['Julia', 'Programming Language', 'Set up']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true

Summary: Julia is a high-level, high-performance, dynamic programming language. It is used for high-performance numerical analysis and computational science.
---

## Installation

[Julia](https://julialang.org) can be installed on MacOS using Homebrew. 

```bash
brew cask install julia
```

## Jupyter kernel

The installation of the Julia kernel for Jupyter is straightforward following this [link](https://github.com/JuliaLang/IJulia.jl/).

On MacOS, from a Terminal, run `julia` to launch a Julia session. Then, run the following commands:
```julia
using Pkg
Pkg.add("IJulia")
```
