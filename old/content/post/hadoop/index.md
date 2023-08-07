---
title: Set up Hadoop
author: ~
date: '2018-12-08'

slug: hadoop
categories: ['Hadoop']
tags: ['Hadoop', 'Set up']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true
  
Summary: Apache Hadoop is an open-source project for reliable, scalable and distributed computing.
---

## Hadoop deployment modes

There are three ways to deploy Hadoop:

* Local mode
* Pseudo-distributed mode
* Distributed mode

## Requisites to the installation

### Java

Check if **Java** is installed:
```bash
java -version
```

It should return something like that:
```bash
java version "1.8.0_***"
Java(TM) SE Runtime Environment (build 1.8.0_***-b11)
Java HotSpot(TM) 64-Bit Server VM (build 25.***-b11, mixed mode)
```

If not, you could go on [java.com](https://www.java.com/fr/download/) and download it.

### SSH

On MacOS, the **Remote Login** must be enable to authorise SSH. It is located in **Systeme Preference** and **Sharing**.

Try to ssh to *localhost* without a passphrase/password. This is important because we do not want to enter a passphrase/password every time Hadoop connect to a node.
```bash
ssh localhost
```

If you can not, run these commands to create a key and put it into the authorised one to connect.
```bash
ssh-keygen -t rsa -P ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

## Set up on MacOs

Let's install Hadoop with Homebrew.
```bash
brew install hadoop
```

As a result, we see where are Hadoop's config files:
```bash
/usr/local/opt/hadoop/libexec/etc/hadoop/hadoop-env.sh
/usr/local/opt/hadoop/libexec/etc/hadoop/mapred-env.sh
/usr/local/opt/hadoop/libexec/etc/hadoop/yarn-env.sh
```

Moreover, the `JAVA_HOME` has been set to the result of the command `/usr/libexec/java_home`.
And finally, the Hadoop's files are in the folder `/usr/local/Cellar/hadoop/*`.
Now, in order to simplify the commands, it is common to add some environment variables to the `.bashrc` file.
```bash
# Hadoop environment
export HADOOP_HOME=/usr/local/Cellar/hadoop/*/libexec
PATH=$HADOOP_HOME/bin:$PATH
```

### Configure HDFS for the Pseudo-Distributed mode

#### Use a single DataNode for each block

Add the following lines to the file `$HADOOP_HOME/etc/hadoop/hdfs-site.xml`.

```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
</configuration>
```

#### Configure the NameNode port

Add the following lines to the file `$HADOOP_HOME/etc/hadoop/core-site.xml`.

```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>
```

#### Set the runtime framework for executing MapReduce jobs

Add the following lines to the file `$HADOOP_HOME/etc/hadoop/mapred-site.xml`.

```xml
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
</configuration>
```

#### Implement the service _mapreduce_shuffle_.
Add the following lines to the file `$HADOOP_HOME/etc/hadoop/yarn-site.xml`.

```xml
<configuration>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
</configuration>
```


#### Format the filesystem

```bash
hdfs namenode -format
```

Then, you can start the NameNode and DataNode deamons.
```bash
$HADOOP_HOME/sbin/start-dfs.sh
```
It is possible to check if it's working using the UI interface: [http://localhost:50070/](http://localhost:50070/) or [http://localhost:9870/](http://localhost:9870/).

And start the Ressource and Node managers.
```bash
$HADOOP_HOME/sbin/start-yarn.sh
```
It is possible to check if it's working using the UI interface: [http://localhost:8088/](http://localhost:8088/).

## Common Errors

### Incompatible clusterIDs

You should reformat the name node with the right clusterId.
```bash
hdfs namenode -format -clusterId CID-...
```

