Notes for [Tutorial on deeplearning.net with theano](http://deeplearning.net/tutorial/contents.html)

# Tutorial

## [Convolutional Neural Networks (LeNet)](http://deeplearning.net/tutorial/lenet.html)

The description of CNN seems a little tedious and hard to understand.


## AutoEncoder

An auto-encoder and corresponding decoder make up a lossy-compression function, by transferring the input to a lower dimension and recover it back.

For example: 

```
f(x) = x//10
g(x) = x*10
```

Will transfer 101 to 10 and recover to 100, there will be some loss, but the encoded (latent) representation is much less in the dimension.

### Denoising autoencoder

> A denoising autoencoders tries to reconstruct the input from a corrupted version of it by projecting it first in a latent space and reprojecting it afterwards back in the input space.

**a corrupted version of it** is the difference compared with the original autoencoder.

I failed to understand:

> To plot our filters we will need the help of `tile_raster_images`

## Stacked Denoising Autoencoders (SdA)

It seems that we are still use the MLP, but with the Stacked Denoising AutoEnoder trained for our initial weight (and bias) configuration.

## Restricted Boltzmann Machines (RBM)

I quickly read through it, have no idea what they are talking about at all. :(

## DBN

> One can also observe that the code for the DBN is very similar with the one for SdA, because both involve the principle of unsupervised layer-wise pre-training followed by supervised fine-tuning as a deep MLP. The main difference is that we use the RBM class instead of the dA class.

## Hybrid Monte-Carlo Sampling

Skipped.

# Installation & Configuration

There is a pretty nice tutorial to [install all the stuff in windows](https://github.com/Theano/Theano/issues/5348)


In order to run with GPU support, with `.theanorc` file:

```
[global]
floatx = float32
device = gpu
mode = FAST_RUN

[dnn]
enabled=False

[lib]
cnmem=0.75
```

We can configure for every single project with its own configure file:

```
set THEANORC=.theanorc
python files_to_run.py
```

And the `activate` file for our conda env:

```
"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"
activate H:\py_env_conda\dl_theano_py2
```

The first line make the build stage available.

It is the `old version` of configuration, newer with guparray for GPU. We failed to enable dnn (Device not support error) even we put the files into the correct files.


# Conclusion

* The code style is not so appealing, and too much tricks involved.
    - Declared shared memory explicitly to make use of GPU
    - Hand calculate number of units in each layer in LeNet example