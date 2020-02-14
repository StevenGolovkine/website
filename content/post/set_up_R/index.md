---
title: Set up R
author: ~
date: '2018-12-05'

slug: set-up-r
categories: ['R']
tags: ['R', 'Set up']

output:
  blogdown::html_page:
    toc: true
    number_sections: false

image:
  caption: ""
  focal_point: Smart
  preview_only: true
  
Summary: R is a software for statistical computing and graphics. It is very easy to install on macOS.
---

## Installation

[**R**](https://cran.r-project.org) is very easy to install on MacOS.

The first thing to do is to add **R** to the available formulae in **Homebrew**. And then, install **R**.

```bash
brew tap homebrew/science
brew install r
```


It could be necessary to install **XQuartz** to use **R** (but it is also possible that it is installed by default)

```bash
brew cask install xquartz
```

A nice GUI to use with **R** is **Rstudio**.

```bash
brew cask install rstudio
```

## Jupyter kernel

The installation of the **R** kernel for Jupyter is straightforward following this [link](https://irkernel.github.io/installation/).

On MacOS, from a Terminal, run `R` to launch a **R** session. Then, run the following commands:
```r
install.packages('IRKernel') # Install the package
IRKernel::installspec() # Make Jupyter to see the R kernel
```

## Configuration of the proxies

How to install packages if you have to deal with proxies? First, you should know the repository where **R** is installed.

```r
R.home()
```

And then, you have to add to the file `${R_HOME}/etc/Renviron` the lines:

```bash
http_proxy=http://<user_name>:<password>@<host>:<port>/
https_proxy=https://<user_name>:<password>@<host>:<port>/
ftp_proxy=ftp://<user_name>:<password>@<host>:<port>/
```

## Some modification to functions

### Summary functions

<details>
<summary>Dataframe summary</summary>
<p>
```{r}
summary_df <- function(df){
  # Function that get a dataframe as input and return a list with two entries.
  # One entry is for factor variables, which is a list that count the factors.
  # The other entry is for numeric variables, which contains statistics on it.
  result <- list()
  
  if(any(sapply(df, class) == 'factor')){
    result$Factor <- df %>% select_if(is.factor) %>% imap(summary_column)
  }
  if(any(sapply(df, class) == 'numeric')){
    result$Numeric <- df %>% select_if(is.numeric) %>% imap_dfr(summary_column)
  }
  
  return(result)
}
```
</p>
</details>	

<details>
<summary>Column dataframe summary</summary>
<p>
```{r}
summary_column <- function(df.column, name.column){
  # Function that get a column from a dataframe and return statistics on it.
  # Depending on the column class, the results will not be the same.
  if(class(df.column) == 'factor'){
    colName <- name.column
    df.column %>% fct_count() %>% rename(!!colName := f, Count = n)    
  } else if(class(df.column) == 'numeric'){
    tibble(
      Name = name.column,
      NA_num = sum(is.na(df.column)),
      Unique = length(unique(df.column)),
      Range = max(df.column, na.rm = TRUE) - min(df.column, na.rm = TRUE),
      Mean = round(mean(df.column, na.rm = TRUE), digits = 2),
      Variance = round(var(df.column, na.rm = TRUE), digits = 2),
      Minimum = min(df.column, na.rm = TRUE),
      Q05 = quantile(df.column, probs = .05, na.rm = TRUE),
      Q10 = quantile(df.column, probs = .10, na.rm = TRUE),
      Q25 = quantile(df.column, probs = .25, na.rm = TRUE),
      Q50 = quantile(df.column, probs = .50, na.rm = TRUE),
      Q75 = quantile(df.column, probs = .75, na.rm = TRUE),
      Q90 = quantile(df.column, probs = .90, na.rm = TRUE),
      Q95 = quantile(df.column, probs = .95, na.rm = TRUE),
      Maximum = max(df.column, na.rm = TRUE)
    )
  }
}
```
</p>
</details>

### Print functions

<details>
<summary>Print summary dataframe</summary>
<p>
```{r}
print_summary_df <- function(l){
  # Print function for the summary of dataframe to be rendered in html.
  for(i in seq_along(l)){
    cat(glue::glue("<ul>"))
    cat(glue::glue("<li> **{names(l)[i]} variables** </li>\n\n"))
    if(class(l[[i]]) == 'list'){
      for(j in seq_along(l[[i]])){
        cat(glue::glue("<ul>"))
        cat(glue::glue("<li> {names(l[[i]][j])} </li>\n\n"))
        cat('<div style="overflow-x:auto;">\n')
        l[[i]][j] %>%
          kable(format = 'html') %>%
          kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), position = "center") %>%
          print()
        cat('</div></ul>\n\n')
      }
    } else{
      cat('<div style="overflow-x:auto;">\n')
      l[[i]] %>%
        kable(format = 'html') %>%
        kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), position = "center") %>%
        print()  
       cat('</div>\n')
    }
    cat(glue::glue("</ul>\n"))
  }
}

```
</p>
</details>

<details>
<summary>Print dataframe</summary>
<p>
```{r}
print_df <- function(l){
  # Print function for dataframe to be renderer in html.
  cat('<div style="overflow-x:auto;">\n')
  l %>% 
    kable(format = 'html') %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), position = "center") %>%
    print()
  cat('</div></ul>\n\n')
}
```
</p>
</details>

<details>
<summary>Print summary lm</summary>
<p>
```{r}
print_summary_lm <- function(lm_summary){
  # Print function for lm summary to be renderer in html.
  cat(glue::glue("Results of the linear model on the **{lm_summary$call$data}** dataset.\n"))
  
  cat(glue::glue("<ul>"))
  # Print the formula
  cat(glue::glue("<li> *Formula*: ", 
                 "{deparse(lm_summary$call$formula)} </li>"))
  
  # Treat the residuals
  cat(glue::glue("<li> *Residuals* </li>\n"))
  cat('<div style="overflow-x:auto;">\n')
  lm_summary$residuals %>% summary_column(name.column = 'Residuals') %>%
    kable(format = 'html', digits = 2) %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), position = "center") %>%
    print()
  cat('</div>\n\n')
  
  # Treat the regression coefficient
  cat(glue::glue("<li> *Coefficients* </li>\n"))
  cat('<div style="overflow-x:auto;">\n')
  coef <- lm_summary$coefficients
  coef[, 'Pr(>|t|)'] <- format.pval(coef[, 'Pr(>|t|)'])
  coef <- coef %>% as.data.frame(stringsAsFactors = FALSE) %>% 
    rownames_to_column('Variable') %>% 
    as.tibble() %>% 
    map_at(c("Estimate", "Std. Error", "t value"), as.numeric)
  coef %>% as.tibble() %>%
    kable(format = 'html', digits = 5) %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), position = "center") %>%
    print()
  cat('</div>\n\n')
  
  # Treat other stats
  pval <- format.pval(pf(lm_summary$fstatistic[1L], lm_summary$fstatistic[2L],
                         lm_summary$fstatistic[3L], lower.tail = FALSE), digits = 3)
  cat(glue::glue("<li> *Residual standard error*: ", 
                 "{round(lm_summary$sigma, 3)} on {lm_summary$df[2]} degrees of freedom. </li>"))
  cat(glue::glue("<li> *Multiple $R^2$*: {round(lm_summary$r.squared, 3)}.</li>"))
  cat(glue::glue("<li> *Adjusted $R^2$*: {round(lm_summary$adj.r.squared, 3)}.</li>"))
  cat(glue::glue("<li> *F-statistic*: ", 
                 "{round(lm_summary$fstatistic[1L], 3)} on {lm_summary$fstatistic[2L]} and {lm_summary$fstatistic[3L]}, ",
                 "p-value: {pval}. </li>"))
  
  cat(glue::glue("</ul>"))
}
```
</p>
</details>

<details>
<summary>Print summary glm</summary>
<p>
```{r}
print_summary_glm <- function(glm_summary){
  cat(glue::glue("Results of the model on the **{glm_summary$call$data}** dataset.\n"))
  
  cat(glue::glue("<ul>"))
  # Print the formula
  cat(glue::glue("<li> *Formula*: ", 
                 "{deparse(glm_summary$call$formula)} </li>"))
  
  # Treat the residuals
  cat(glue::glue("<li> *Residuals* </li>\n"))
  cat('<div style="overflow-x:auto;">\n')
  glm_summary$deviance.resid %>% summary_column(name.column = 'Residuals') %>%
    kable(format = 'html', digits = 2) %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), position = "center") %>%
    print()
  cat('</div>\n\n')
  
  # Treat the regression coefficient
  cat(glue::glue("<li> *Coefficients* </li>\n"))
  cat('<div style="overflow-x:auto;">\n')
  coef <- glm_summary$coefficients
  coef[, 'Pr(>|z|)'] <- format.pval(coef[, 'Pr(>|z|)'])
  coef <- coef %>% as.data.frame(stringsAsFactors = FALSE) %>% 
    rownames_to_column('Variable') %>% 
    as.tibble() %>% 
    map_at(c("Estimate", "Std. Error", "z value"), as.numeric)
  coef %>% as.tibble() %>%
    kable(format = 'html', digits = 5) %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), position = "center") %>%
    print()
  cat('</div>\n\n')
  
  # Treat other stats
  cat(glue::glue("<li> *Null deviance*: ", 
                 "{round(glm_summary$null.deviance, 3)} on {glm_summary$df.null} degrees of freedom. </li>"))
  cat(glue::glue("<li> *Residual deviance*: ", 
                 "{round(glm_summary$deviance, 3)} on {glm_summary$df.residual} degrees of freedom. </li>"))
  cat(glue::glue("<li> *AIC*: ", 
                 "{round(glm_summary$aic, 3)}</li>"))
  
  cat(glue::glue("</ul>\n"))
}

```
</p>
</details>

### Plot functions

<details>
<summary>Plot confusion matrix</summary>
<p>
```{r}
plot_confusion_matrix <- function(confusion_matrix){
  confusion_matrix %>%
    as.data.frame(optional = TRUE) %>% 
    rownames_to_column() %>%
    rename('Var1' = '.') %>%
    ggplot() +
    geom_text(aes(x = Var1, y = Var2, label = Freq), size = 4) +
    xlab('Prediction') +
    ylab('True') +
    geom_hline(aes(yintercept = 1.5), size = 0.2) +
    geom_vline(aes(xintercept = 1.5), size = 0.2) +
    theme_bw() +
    scale_x_discrete(position = "top") +
    theme(panel.grid = element_blank(),
          axis.ticks = element_blank())
}
```

</p>
</details>

<details>
<summary>Plot regsubset summary</summary>
<p>
```{r}
ggregsubsets <- function(x, criterion = c("rsq", "rss", "adjr2", "cp", "bic")){
  # https://gist.github.com/dkahle/7942a7eba8aaa026d0bab6a1e9d88580
  require(dplyr); require(ggplot2); require(tidyr)
  if(inherits(x, "regsubsets")) x <- summary(x)
  if(!inherits(x, "summary.regsubsets"))
    stop("The input to ggregsubsets() should be the result of regsubsets().")
  df <- bind_cols(
    as.data.frame(x$which), 
    as.data.frame(x[criterion]),
    data.frame(nvars = 1:nrow(x$which))
  )
  names(df)[1] <- "Int"
  if("rsq" %in% criterion) df <- df %>% mutate(rsq = 100*rsq)
  if("adjr2" %in% criterion) df <- df %>% mutate(adjr2 = 100*adjr2)
  
  df <- df %>% 
    gather(variable, is_in, -criterion, -nvars) %>% 
    gather(measure, value, -nvars, -variable, -is_in)
  
  if("rsq" %in% criterion) df[df['measure'] == 'rsq', 'measure'] <- '$R^2$'
  if("rss" %in% criterion) df[df['measure'] == 'rss', 'measure'] <- '$RSS$'
  if("adjr2" %in% criterion) df[df['measure'] == 'adjr2', 'measure'] <- 'Adjusted $R^2$'
  if("cp" %in% criterion) df[df['measure'] == 'cp', 'measure'] <- '$C_p$'
  if("bic" %in% criterion) df[df['measure'] == 'bic', 'measure'] <- '$BIC$'
  
  p <- ggplot(df, aes(variable, factor(round(value)))) +
      geom_tile(aes(fill = is_in)) +
      facet_wrap(~ measure, scales = "free") +
      scale_fill_manual("", values = c("TRUE" = "black", "FALSE" = "white"), guide = FALSE) +
      labs(x = "", y = "")
  return(p)
}
```
</p>
</details>

<details>
<summary>Plot criteria for model selection</summary>
<p>
```{r}
ggcriteria <- function(x, criterion = "bic"){
  require(dplyr); require(ggplot2); require(tidyr)
  if(inherits(x, "regsubsets")) x <- summary(x)
  if(!inherits(x, "summary.regsubsets"))
    stop("The input to ggregsubsets() should be the result of regsubsets().")
  
  if("rsq" == criterion) crit <- '$R^2$'
  if("rss" == criterion) crit <- '$RSS$'
  if("adjr2" == criterion) crit <- 'Adjusted $R^2$'
  if("cp" == criterion) crit <- '$C_p$'
  if("bic" == criterion) crit <- '$BIC$'
  
  if((criterion == "adjr2") | (criterion == "rsq")) m <- which.max(x[[criterion]])
  else m <- which.min(x[[criterion]])
  
  p <- ggplot() +
    geom_line(aes(x = seq(1, length(x[[criterion]])), y = x[[criterion]])) + 
    geom_point(aes(x = m, y = x[[criterion]][m]), col = 'red', size = 3) +
    xlab('Number of variables') +
    scale_x_continuous(breaks = seq(1, length(x[[criterion]])), minor_breaks = NULL) +
    ylab(crit)
  return(p)
}
```
</p>
</details>

<details>
<summary>Plot cross-validation error from `cv.glmnet` function</summary>
<p>
```{r}
ggcv.glmnet <- function(x){
  require(dplyr); require(ggplot2); require(glmnet)
  if(!inherits(x, "cv.glmnet"))
    stop("The input of ggcv.glmnet() should be the result og cv.glmnet().")
  
  df <- tibble(lambda = log(x$lambda), cvm = x$cvm, cvsd = x$cvsd)
  p <- ggplot(df, aes(lambda, cvm, ymin = cvm - cvsd, ymax = cvm + cvsd)) +
    geom_point(col = 'red') +
    geom_errorbar() +
    geom_vline(aes(xintercept = df$lambda[which.min(df$cvm)]), col = 'blue', linetype = 2) +
    xlab('$\\log(\\lambda)$') +
    ylab('Mean-Squared Error')
  return(p)
}
```
</p>
</details>
