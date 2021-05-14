#  A custom loss function out of cross entropy

The cross-entropy loss function is one of the most used loss functions for training deep learning architectures. The cross-entropy loss has its roots in information theory, and it is a simple dot product. The dot product consists of the neural network output, which is a vector of probabilities representing each class by the target vector, which can be a one-hot encoded vector or not. It can be shown as:

<img src="https://latex.codecogs.com/gif.latex?-\sum_{i}&space;y_{i}log(\hat{y})" title="-\sum_{i} y_{i}log(\hat{y})" />

Based on empirical experiments, we have devised a custom loss function devised from the cross-entropy. It has three additional terms, one multiplicand, and two additive terms. It can be represented as:

<img src="https://latex.codecogs.com/gif.latex?-\left&space;(&space;\sum_{i}&space;y{_{i}}^{2}log(\hat{y})&plus;y_{i}&space;-&space;\hat{y}&space;\right)" title="-\left ( \sum_{i} y{_{i}}^{2}log(\hat{y})+y_{i} - \hat{y} \right)" />

Our experiments with the modified loss function perform better in the reinforcement learning settings, and equally performant in supervised and self-supervised settings especially in early stopping scenarios. 


In this repo, we provide three open examples provided by Keras library:

1. Targeting OpenAI Gym cart-pole with an actor-critic architecture
2. Supervised learning of MNIST dataset
3. English to French translation with a seq2seq architecture. 

The cart-pole agent can solve around 2x faster with the custom loss function with seldom instability in training. The MNIST supervised learning example finds slightly better generalization with the custom loss function compared to the cross-entropy loss. Finally, both loss functions perform equally good in lstm sequence-to-sequence architecture. 

#### Cartpole result
![alt text](https://github.com/ggaamm/ce_custom_loss/blob/main/images/custom_loss_cartpole.png "Mean score over last 10 episodes")

Above is the mean score over last 10 episodes for the custom loss, solves the environment around 150 episodes
 
![alt text](https://github.com/ggaamm/ce_custom_loss/blob/main/images/cross_entropy_cartpole.png "Mean score over last 10 episodes")

Above is the mean score over last 10 episodes for the cross entropy loss, solves the environment around 320 episodes

#### Custom loss function code
Below is the custom loss function in python that works with Keras library
```python
def ce_custom_loss(self, y_true, y_pred):
    #print("y_true", y_true, "y_pred", y_pred)
    val = tf.square(y_true) * tf.log(y_pred) + y_true - y_pred
    return tf.reduce_sum(-val, -1)
```

**We are encouraging the readers to try out custom loss function with their own architectures and share their results and experience.**


## Cite this work
You can cite this work in your work as:

It has a DOI:

![Alt text](DOI of this repo for citations)
<img src="https://zenodo.org/badge/doi/10.5281/zenodo.4763263.svgg">

https://zenodo.org/badge/353324836.svg

@misc{Malazgirt2021,
  author = {Malazgirt, G.A.},
  title = {Alternative loss function to the cross-entropy loss for training deep neural networks},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ggaamm/ce_custom_loss}},
  commit = {d14de4ce3f447a1ff42326cbd945b24d08f4a181}
}
