---
title: knitr
author: ~
date: '2019-07-25'

slug: knitr
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

Summary: The package `knitr` is a **R** package for report generation. 
---

The package `knitr` has been created by Yihui Xie. It is disponible on the [CRAN](https://cran.r-project.org/web/packages/knitr/index.html). Check out the official [website](https://yihui.name/knitr/) for more information. It is designed to build and generate nice report in **R**.

## Questions

### How to render Latex formula in ggplot graphs in a report in html?

For a report in pdf, it is quite easy to render Latex formula in a ggplot graph. Just set the `dev` variable to `'tikz'` in a **R** chunk. However, this method produces a pdf of the picture, and some browsers seem to have some trouble to show pdf files. So, the idea is to convert the pdf of the picture into a png file. For that, the **R** chunk accept the option `fig.process` and we will modify it to solve our problem.

```{r eval=FALSE}
fig.process <- function(x) {
    x <- paste0('./', x)
    
    if(stringr::str_detect(x, 'pdf')){
      y <- stringr::str_replace(x, 'pdf', 'png')
      png::writePNG(pdftools::pdf_render_page(x), target = y, dpi = 300)
      return(y)
    } else {
      return(x)
    }
}
```

The function `fig.process` takes a string as input (the path of the picture). If the picture is in pdf, it will convert it into png, otherwise, it will do nothing. And finally, the browser will render the picture correctly.

