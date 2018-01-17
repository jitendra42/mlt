# mlt
Machine Learning Container Tool

`mlt` aids the creation of containers for machine learning jobs.
It does so by making it ease to use container and kubernetes object templates.

![MLT flow diagram](docs/mlt.png)

We have found that a it can be useful to share templates for how to use certain underlying machine learning frameworks. Some have native support from Kubernetes operators, such as mxnet and TensorFlow. Others do not, but still have best practices for how to run on a Kubernetes cluster.

On top of getting boiler plate code at the beginning of a project to work, the best practices may change over time. mlt strive to enable a workflow that allows existing projects to adapt to these without having to reset and start over.

Another aspect of the application development with containers `mlt` address, is _iterative_ container creation. Storage and container creation is supposed to be fast - so why not rebuild containers automatically?
`mlt` has a `--watch` option, which lets you write code and have an IDE like experience.

## Build

Prerequisites:
- git
- python
- virtualenv

Installation:

```bash
$ make
$ . ./env/bin/activate
$ . ./path.sh
# cd outside of the mlt tree
$ cd ..
$ mlt init my-app
$ cd my-app
$ mlt deploy
```

## Usage summary

_TODO_
