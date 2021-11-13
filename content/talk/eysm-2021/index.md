---
title: 'EYSM 2021'
event: 22nd European Young Statisticians Meeting
event_url: https://www.eysm2021.panteion.gr

location: Virtual Conference
address:
  city: 
  postcode: 
  country: 

summary: Clustering multivariate functional data using unsupervised binary trees.

abstract: "We propose a model-based clustering algorithm for a general class of
functional data for which the components could be curves or images. The
random functional data realizations could be measured with error at discrete,
and possibly random, points in the definition domain. Based on Fraiman *et al.* (2013), the idea is to build a set of binary trees by recursive splitting of the observations. At each node of the tree, a model selection test is performed, after expanding the multivariate functional data into a well chosen basis. We consider the Multivariate Functional Principal Component basis, developed in Happ and Greven (2018). Similarly to Pelleg and Moore (2000), using the Bayesian Information Criterion, we test whether there is evidence that the data structure is a mixture model or not at each node of the tree. The number of groups are determined in a data-driven way and does not have to be pre-specified before the construction of the tree. Moreover, the tree structure allows us to consider only a small number of basis functions at each node. The new algorithm provides easily interpretable results and fast predictions for online data sets. Results on simulated datasets reveal good performance in various complex settings. The methodology is applied to the analysis of vehicle trajectories on a German roundabout. The open-source implementation of the algorithm can be accessed [here](https://github.com/StevenGolovkine/FDApy). Complete version of the work is available [here](https://arxiv.org/abs/2012.05973v2}{arxiv:2012.05973)."

# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: "2021-09-06T11:00:00Z"
#date_end: "2019-06-03T11:30:00Z"
all_day: false

# Schedule page publish date (NOT talk date).
#publishDate: "2017-01-01T00:00:00Z"

authors: ['Steven Golovkine', 'Nicolas Klutchnikoff', 'Valentin Patilea']
tags: []

# Is this a featured talk? (true/false)
featured: false

image:
  caption: ''
  focal_point: Right

links: []
url_code: ""
url_poster: ""
url_pdf: "https://www.eysm2021.panteion.gr/files/Proceedings_EYSM_2021.pdf#page=28"
url_slides: "/files/presentation_eysm_2021.pdf"
url_video: ""

# Markdown Slides (optional).
#   Associate this talk with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []

# Enable math on this page?
math: true
---
