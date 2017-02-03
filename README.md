# *Neural-Image-Colorization* using adversarial networks

<img src="" width="640px" align="right">

This is a Tensorflow implementation of *[Image-to-Image Translation with Conditional Adversarial Networks]()* that aims to infer a mapping from X to Y, where X is a single channel "black and white" image and Y is 3-channel "colorized" version of that image.

We make use of [Generative Adversarial Networks]() conditioned on the input to teach a generative neural network how to produce our desired results.

The purpose of this repository is to port the model over to TensorFlow.

## Results

<table style="width:100%">
  <tr>
    <th>Input</th> 
    <th>Output</th>
    <th>Ground-Truth</th>
  </tr>
  <tr>
    <td><img src="lib/" width="100%"></td>
    <td><img src="lib/" width=100%"></td> 
    <td><img src="lib/" width=100%"></td> 
  </tr>
  <tr>
    <td><img src="lib/" width="100%"></td>
    <td><img src="lib/" width="100%"></td> 
    <td><img src="lib/" width=100%"></td> 
  </tr>
  <tr>
    <td><img src="lib/" width="100%"></td>
    <td><img src="lib/" width="100%"></td> 
    <td><img src="lib/" width=100%"></td> 
  </tr>
</table>

## Prerequisites

* [Python 3.5](https://www.python.org/downloads/release/python-350/)
* [TensorFlow](https://www.tensorflow.org/) (>= r0.12)

## Usage

```sh
python train.py
```

```sh
python test.py 'path/to/input/image'
```

## Files

* [colorize.py](./src/colorize.py)

    Main script that interprets the user's desired actions through parsed arguments. 
    
* [generator.py](./src/generator.py)
    
    Contains the generative net that can colorize single-channel images when trained.
    
* [discriminator.py](./src/discriminator.py)
    
    Contains the discriminative net that can discriminate between synthesized colorized images and ground-truth images.
    
* [net.py](./src/net.py)
    
    Contains the neural network super class with universal layer construction and instance normalization methods. 
    
* [train.py](./src/train.py)
    
    Contains a Trainer class that is responsible for training the generative adversarial networks and any related routines such as retrieving training data.