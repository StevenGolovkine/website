---
title: Library Matplotlib
author: ~
date: '2018-12-06'

slug: library-matplotlib
categories: ['Python']
tags: ['Python', 'Library']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true

Summary: Matplotlib is a Python 2D plotting library.
---

[Matplotlib](http://matplotlib.org) is a Python 2D plotting library.

## Use another font

First, you need to download/get back the font in the `ttf` format. Once it is done, you have to find out the location of the matplotlib library.

```bash
python -c "import matplotlib; print(matplotlib.matplotlib_fname())"
```

This folder will be different for everyone depending on the installation but it should ended by `/matplotlib/mpl-data/matplotlibrc`.

Then, you copy the font into the folder `/matplotlib/mpl-data/fonts/ttf/`.
```bash
cp font.ttf ./matplotlib/mpl-data/fonts/ttf/
```

Finally, remove the font cache:
```bash
rm ~/.matplotlib/fontList.py3k.cache
```

Now, you can use the new font:
```python
import matplotlib.pyplot as plt
plt.rc('font', family='font_name')
```
