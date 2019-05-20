# TensorFlow Self-Organizing Map
An implementation of the Kohonen self-organizing map<sup>1</sup> for TensorFlow 1.5 and Python 3.6. This was initially based
off of [Sachin Joglekar's](https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/)
code but has a few key modifications:
 * Uses TensorFlow broadcasting semantics instead of `tf.pack` and `for` loops.
 * Input data is expected from a `Tensor` rather than a `tf.placeholder`, allowing for use with faster and more complex input data pipelines.
 * Training uses the batch algorithm rather than the online one, providing a major speed boost if you have the GPU RAM.
 Also, as a result of that, I added...
 * Multi-GPU support (for single machines with multiple GPUs, it doesn't have multi-node training).
 * Some summary operations for Tensorboard visualization

 `example.py` contains a simple example of its usage by training a SOM on a 3 cluster toy dataset. The resulting
 u-matrix should look something like this:

 ![Example U-Matrix](https://github.com/cgorman/tensorflow-som/blob/master/example_umatrix.png)
 
Note that the example requires scikit-leran to be installed.

 I was going to write a blog post about this but I ended up just repeating everything I wrote in the comments,
 so please read them if you'd like to understand the code. For reference, the batch formula for SOMs is
 
 ![SOM batch formula](https://github.com/cgorman/tensorflow-som/blob/master/batch_formula.gif)
 
 where theta is the neighborhood function and x is the input vector.
 
 The activity function turns the distance between each of the weights and an input vector into a value between 0 and 1, i.e. similar weights elicit a higher activity.
 The activity function is parameterized with the `output_sensitivity` variable.
 When this value is close to zero the range of distances that elicit high activity is wider, and vice versa.
 Here is an example of a few different values of the output sensitivity (`-c` here):
 
 ![Effect of Output Sensitivity Parameter](https://github.com/cgorman/tensorflow-som/blob/master/output_sens.png)
 
## Note about the learning rate
There was a really dumb bug in commits before 2a0ee25 where the learning rate (alpha) was set extremely incorrectly.
Instead of shrinking from n to 0, it grew from n to 1.
If you had bad luck with this module previously, this may fix it.
Sorry for the goof!

 <sup>1</sup>http://ieeexplore.ieee.org/document/58325/
