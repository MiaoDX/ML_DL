Notes for [Tutorial on deeplearning.net with theano](http://deeplearning.net/tutorial/contents.html)


# [Convolutional Neural Networks (LeNet)](http://deeplearning.net/tutorial/lenet.html)

The description of CNN seems a little tedious and hard to understand.

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