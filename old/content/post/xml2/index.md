---
title: Package xml2
author: ~
date: '2018-12-06'

slug: package-xml2
categories: ['R']
tags: ['R', 'Package']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true

Summary: The package `xml2` is a **R** package for parsing XML files.
---

The package `xml2` has been created by Hadley Wickham among others. It is disponible on the [CRAN](https://cran.r-project.org/web/packages/xml2/index.html). It is designed to work with XML files in **R**.

## Installation issues

### Configuration failed because libxml-2.0 was not found

First, check that the folders given by the following commands exits:

```bash
which xml2-config
# /usr/bin/xml2-config

xml2-config --libs
# -L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/usr/lib -lxml2 -lz -lpthread -licucore -lm

xml2-config --cflags
# -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/usr/include/libxml2

which pkg-config
# /usr/local/bin/pkg-config

pkg-config --cflags libxml-2.0
# -I/usr/include/libxml2

pkg-config --libs libxml-2.0
# -lxml2
```

In the case where, `/usr/include` does not exists, run the command:

```
xcode-select --install
```

And then, re-try to install the package `xml2`.
