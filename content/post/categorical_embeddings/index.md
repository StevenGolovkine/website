---
title: Categorical Embeddings
author: ~
date: '2020-04-20'

slug: categorical-embeddings
categories: ['Deep Learning', 'Python']
tags: ['Deep Learning', 'Python', 'Keras']

output:
  blogdown::html_page:
    toc: true
    number_sections: false

image:
  caption: Photo by Kelly Sikkema on Unsplash
  focal_point: Smart

Summary: An introduction to categorical embeddings using `Keras`.
---

This post is based on the Deep Learning course from the Master Datascience Paris Saclay. Materials of the course can be found [here](https://github.com/m2dsupsdlclass/lectures-labs). The complete code can be found on a Kaggle [kernel](https://www.kaggle.com/stevengolo/categorical-embeddings).

**Goal of the notebook**

* Introduction to categorial embeddings using Keras.

We will use the embeddings through the whole lab. They are simply represented by a matrix of tunable (weights). Let us assume that we are given a pre-trained embedding matrix for a vocabulary of size $10$. Each embedding vector in that matrix has dimension $4$. Those dimensions are too small to be realistic and are only used for demonstration purposes.

```python
# Define an embedding matrix
EMBEDDING_SIZE = 4
VOCAB_SIZE = 10

embedding_matrix = np.arange(EMBEDDING_SIZE * VOCAB_SIZE, dtype='float32')
embedding_matrix = embedding_matrix.reshape(VOCAB_SIZE, EMBEDDING_SIZE)
```

    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]
     [12. 13. 14. 15.]
     [16. 17. 18. 19.]
     [20. 21. 22. 23.]
     [24. 25. 26. 27.]
     [28. 29. 30. 31.]
     [32. 33. 34. 35.]
     [36. 37. 38. 39.]]


To access the embedding for a given integer (ordinal) symbol $i$, one may either:

* simply index (slice) the embedding matrix by $i$, using numpy integer indexing


```python
idx = 3
embedding_matrix[idx]
```

    [12. 13. 14. 15.]


* compute a one-hot encoding vector $\mathbf{v}$ of $i$, then compute a dot product with the embedding matrix


```python
def onehot_encode(dim, label):
    return np.eye(dim)[label]

onehot_idx = onehot_encode(VOCAB_SIZE, idx)
```

    [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]



```python
np.dot(onehot_idx, embedding_matrix)
```

    [12. 13. 14. 15.]


## The Embedding layer in Keras

In `Keras`, embeddings have an extra parameter, `input_length` which is typically used when having a sequence of symbols as input (sequence of words for example). Here, the length will always be 1.

    Embedding(output_dim=embedding_size, 
              input_dim=vocab_size, 
              input_length=sequence_length, 
              name='my_embedding')
    
Futhermore, we load the fixed weights from the previous matrix instead of using a random initialization.

    Embedding(output_dim=embedding_size, 
              input_dim=vocab_size, 
              weights=[embedding_matrix], 
              input_length=sequence_length, 
              name='my_embedding')


```python
# Define an embedding layer
embedding_layer = Embedding(output_dim=EMBEDDING_SIZE, 
                            input_dim=VOCAB_SIZE, 
                            weights=[embedding_matrix], 
                            input_length=1, 
                            name='My_embedding')
```


```python
# Define a Keras model using this embedding
x = Input(shape=[1], name='Input')
embedding = embedding_layer(x)
model = Model(inputs=x, outputs=embedding)
```

The output of an elbedding layer is then a 3-dimensional tensor of shape `(batch_size, sequence_length, embedding_size)`. `None` is a marker for dynamic dimensions.


```python
model.output_shape
```

    (None, 1, 4)

The embedding weights can be retrieved as model parameters.


```python
model.get_weights()
```

    [array([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.],
            [16., 17., 18., 19.],
            [20., 21., 22., 23.],
            [24., 25., 26., 27.],
            [28., 29., 30., 31.],
            [32., 33., 34., 35.],
            [36., 37., 38., 39.]], dtype=float32)]


The `model.summary()` method gives the list of trainable parameters per layer in the model.


```python
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Input (InputLayer)           [(None, 1)]               0         
    _________________________________________________________________
    My_embedding (Embedding)     (None, 1, 4)              40        
    =================================================================
    Total params: 40
    Trainable params: 40
    Non-trainable params: 0
    _________________________________________________________________


We can use the `predict` method of the Keras embedding model to project a single integer label into the matching embedding vector.


```python
labels_to_encode = np.array([[3]])
model.predict(labels_to_encode)
```

    array([[[12., 13., 14., 15.]]], dtype=float32)

We can do the same with batch of integers.


```python
labels_to_encode = np.array([[3], [3], [0], [9]])
model.predict(labels_to_encode)
```

    array([[[12., 13., 14., 15.]],
    
           [[12., 13., 14., 15.]],
    
           [[ 0.,  1.,  2.,  3.]],
    
           [[36., 37., 38., 39.]]], dtype=float32)


The output of an embedding layer is then a 3-dimensional tensor of shape `batch_size, sequence_length, embedding_size`. In order to remove the sequence dimension, useless here, we use the `Flatten()` layer.


```python
x = Input(shape=[1], name='Input')
y = Flatten()(embedding_layer(x))
model2 = Model(inputs=x, outputs=y)
```


```python
model2.output_shape
```




    (None, 4)




```python
model2.predict(np.array([3]))
```




    array([[12., 13., 14., 15.]], dtype=float32)



There are $40$ trainable parameters for the `model2` (which is the same as `model`).


```python
model2.summary()
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Input (InputLayer)           [(None, 1)]               0         
    _________________________________________________________________
    My_embedding (Embedding)     (None, 1, 4)              40        
    _________________________________________________________________
    flatten (Flatten)            (None, 4)                 0         
    =================================================================
    Total params: 40
    Trainable params: 40
    Non-trainable params: 0
    _________________________________________________________________


Note that we re-used the same `embedding_layer` instance in both `model` and `model2`: therefore, the two models share exactly the same weights in memory.


```python
model2.set_weights([np.ones(shape=(VOCAB_SIZE, EMBEDDING_SIZE))])
```


```python
labels_to_encode = np.array([[3]])
model2.predict(labels_to_encode)
```




    array([[1., 1., 1., 1.]], dtype=float32)




```python
model.predict(labels_to_encode)
```




    array([[[1., 1., 1., 1.]]], dtype=float32)



The previous model definitions used the [function API of Keras](https://keras.io/getting-started/functional-api-guide/). Because the embedding and flatten layers are just stacked one after the other it is possible to instead use the [Sequential model API](https://keras.io/getting-started/sequential-model-guide/).

Let defined a third model named model3 using the sequential API and that also reuses the same embedding layer to share parameters with model and model2.


```python
model3 = Sequential([embedding_layer, Flatten()])
```


```python
model3.predict(labels_to_encode)
```




    array([[1., 1., 1., 1.]], dtype=float32)


