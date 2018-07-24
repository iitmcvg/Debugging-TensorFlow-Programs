## Debugging TensorFlow Programs

---

Before seeing why debugging tensorflow program is difficult, lets see a sample program written in TensorFlow.

+++

```python
import numpy as np
import tensorflow as tf

x_data = np.linspace(-20,20, 100)
y_data = 5*x_data + 7 + np.random.randn(len(x_data))*0.05


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

global_step_tensor = tf.train.get_or_create_global_step()


W = tf.Variable(tf.random_uniform(shape = [],minval = 0,maxval = 100))
b = tf.Variable(tf.random_uniform(shape = [],minval = -100,maxval = 100))

yPred = W*x + b

loss = tf.reduce_sum(tf.square(y - yPred))
tf.summary.scalar('Loss', loss)
tf.summary.scalar('weight', W)
tf.summary.scalar('bias', b)


LEARNING_RATE = 0.0000015
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step_tensor)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train_logs',sess.graph)
while True:
    _, summary, step, l2loss = sess.run([train_step, merged, global_step_tensor, loss],feed_dict={x:x_data, y:y_data })
    train_writer.add_summary(summary, step)
    print ("Iteration: " + str(step) + " Loss: " + str(l2loss))
```
@[1-2](Necessary libraries are imported)
@[4-5](Synthetic Input data is created)
@[11](Creating the global step tensor, where we will store the amount of iterations passed in the graph)
@[14-15](Initialising the weights and biases)
@[17](Defining our linear model)
@[19](Defining our Loss function (L2 Difference)
@[20-22](Defining Tensorboard summary Scalars)
@[25-26](Learning Rate and Optimizers are defined (Gradient Descent Optimization)
@[28-30](Creating **Tensorflow Session** and running initialization operation)
@[33-38](Training Loop)

---

In the above program, what will happen if I write 

```python
# Lets print the weights in linear regression model
print (W)
```

Whats the output you expect?

+++

```
<tf.Variable 'Variable:0' shape=() dtype=float32_ref>
```

This is not the number we expected right?

---

In tensorflow we construct a `**Computational Graph**` when we write a program. Then we use `tf.Session()` to get values form the computational graph.

For example, in the above linear regression model, the computational graph is like this

![Linear_regression_graph](/images/linear_regression_graph.png)

---
