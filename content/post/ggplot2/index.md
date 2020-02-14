---
title: Package ggplot2
author: ~
date: '2018-12-19'

slug: package-ggplot2
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

Summary: The package `ggplot2` is a **R** package for data vizualisation. It is part of the tidyverse.
---

The package `ggplot2` has been created by Hadley Wickham among others. It is disponible on the [CRAN](https://cloud.r-project.org/web/packages/ggplot2/index.html). Check out the official [website](https://cloud.r-project.org/web/packages/ggplot2/index.html) for more information. It is designed to provide nice graphics in an quite easy way.

## Use custom theme

First, define a particular function that define a new theme based on another.
```{r eval=FALSE}
theme_custom = function(base_family = "Times"){
  theme_minimal(base_family = base_family) %+replace%
    theme(
      plot.title = element_text(size = 20),
      plot.subtitle = element_text(size = 16, vjust = -1),
      
      axis.title = element_text(size = 18),
      axis.text = element_text(size = 16),
      
      axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0), angle = 90),
      axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 20, l = 0)),
      
      strip.text.x = element_text(size = 16),
      strip.text.y = element_text(size = 16),
      
      legend.text = element_text(size = 18),
      legend.text.align = 0
    )
}
```


Then, just use it as any other theme.
```{r eval=FALSE, message=FALSE, warning=FALSE, paged.print=FALSE}
ggplot(aes(x, y)) + theme_custom()
```

