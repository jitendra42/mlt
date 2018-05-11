# Distributed TensorFlow MLT Application

A distributed TensorFlow MNIST model which designates worker 0 as the chief.

This template requires that the TFJob Operator is installed on your
cluster.  The command below shows an example of how to verify if TFJob
is installed:

```bash
$ kubectl get crd | grep tfjob
tfjobs.kubeflow.org               1d
```

If TFJob is not installed on your cluster, see the installation
instructions [here](https://github.com/kubeflow/tf-operator#installing-the-tfjob-crd-and-operator-on-your-k8s-cluster).
