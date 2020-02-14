---
title: MySQL
author: ~
date: '2018-12-06'

slug: mysql
categories: ['Software']
tags: ['Software', 'Set up']

output:
  blogdown::html_page:
    toc: true
    number_sections: false
    
image:
  caption: ""
  focal_point: Smart
  preview_only: true
  
Summary: MySQL is an open-source relational database management system.
---


[MySQL](https://www.mysql.com/) is an open-source relational database management system.

## Installation and configurations

We simply use **Homebrew** to install **MySQL** under MacOS.

```bash
brew install mysql
```

Then, we can launch **MySQL** by running the command:
```bash
brew services start mysql
# ==> Successfully started 'mysql' (label: homebrew.mxcl.mysql)
```

One can recommended to set a password for the _root_ user and only authorize the access from _localhost_.

```bash
mysql_secure_installation
```

And then, we re-launch **MySQL**.

```bash
brew services restart mysql
```

## Connection

We connect to **MySQL** by running the commands:
```bash
mysql --host=localhost --user=root -p
# Enter password : ****
```

It is not recommended to work in _root_ on databases because this user has all privileges. One could create a restrictive user of the database.

```sql
GRANT ALL PRIVILEGES ON nom_base.* TO 'name'@'localhost' IDENTIFIED BY 'password';
```

One can save the database into a file using the command:

```bash
mysqldump -u user -p --opt database_name > save.sql
```

## Configuration

In order to use every possible characters into the string in the database, one should activate the UTF-8 encode.

```bash
mysql --host=localhost --user=name -p --default_character-set=utf8
# Enter password : ****
```

In order to insert data into the database using external files, one should activate this possibility.

```bash
mysql -h localhost -u name -p --enable-local-infile
# Enter password : ****
```
