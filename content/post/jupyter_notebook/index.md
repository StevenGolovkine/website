---
title: Jupyter Notebook
author: ~
date: '2018-12-07'

slug: jupyter-notebook
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

summary: Jupyter is a programming software which use notebooks widely used in data science.
---

## Update IRkernel after updating R

Modify the file `kernel.json` into the folder `~/Library/Jupyter/kernels/ir`.
Replace the value of the argument `argv` by the new path of R.
