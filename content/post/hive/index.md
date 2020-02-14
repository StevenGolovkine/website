---
title: Set up Hive
author: ~
date: '2018-12-09'

slug: set-up-hive
categories: ['Hive']
tags: ['Hive', 'Set up']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true
Summary: Apache Hive is a data warehouse software that facilitates the manipulation of large distributed datasets using SQL. 
---

## Set up on MacOs

Let's install **Hive** with **Homebrew.**

```bash
brew install hive
```

The **Hive**'s files are in the folder `/usr/local/Cellar/hive/*`.

Same as **Hadoop**, add some environment variables to the `.bashrc`file.

```bash
# Hive environment
export HIVE_HOME=/usr/local/Cellar/hive/*/libexec
PATH=$HIVE_HOME/bin:$PATH
```
In order to launch Hive, Hadoop ressources must be set up.
```bash
# Setup Hadoop
$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh
# Launch Hive
hive
```

## Common errors

### Metastore troubles

Just remove and reload the metastore.
```bash
rm -rf metastore_db derby.log
$HIVE_HOME/bin/schematool -initSchema -dbType derby
```
