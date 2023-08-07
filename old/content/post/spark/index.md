---
title: Set up Spark
author: ~
date: '2018-12-09'

slug: set-up-spark
categories: ['Spark']
tags: ['Spark', 'Set up']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true

Summary: Apache Spark is a software for large-scale data processing.
---


## Set up on MacOs

Let's install **Spark** with **Homebrew.**

```bash
brew install apache-spark
```

The **Spark**'s files are in the folder `/usr/local/Cellar/apache-spark/*`.

Same as **Hadoop**, add some environment variables to the `.bashrc`file.

```bash
# Spark environment
export SPARK_HOME=/usr/local/Cellar/apache-spark/*/libexec
PATH=$SPARK_HOME/bin:$PATH
```

## Configure Spark to run with YARN

Edit the file `$SPARK_HOME/conf/spark-env.sh.template` by adding the following line:
```bash
HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
```
And then rename it as `spark-env.sh`.

Finally, you can run the Spark command lines on YARN with the command:
```bash
spark-shell --master yarn --deploy-mode client
```
