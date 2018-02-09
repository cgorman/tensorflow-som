# TensorFlow Self-Organizing Map
An implementation of the Kohonen self-organizing map<sup>1</sup> for TensorFlow 1.5 and Python 3.6. This was initially based
off of [Sachin Joglekar's](https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/)
code but has a few key modifications:
 * Uses TensorFlow broadcasting semantics instead of `tf.pack` and `for` loops.
 * Input data is expected from a `Tensor` rather than a `tf.placeholder`, allowing for use with faster and more complex
 input data pipelines.
 * Training uses the batch algorithm rather than the online one, providing a major speed boost if you have the GPU RAM.
 Also, as a result of that, I added...
 * Multi-GPU support (for single machines with multiple GPUs, it doesn't have multi-node training).
 * Some summary operations for Tensorboard visualization

 `example.py` contains a simple example of its usage by training a SOM on a 3 cluster toy dataset. The resulting
 u-matrix should look something like this:

 ![Example U-Matrix](https://github.com/cgorman/tensorflow-som/blob/master/example_umatrix.png)
 
 <sup>1</sup>http://ieeexplore.ieee.org/document/58325/
