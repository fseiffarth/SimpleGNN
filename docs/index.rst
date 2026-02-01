.. SimpleGNN documentation master file

SimpleGNN Documentation
=======================

Welcome to SimpleGNN's documentation!

SimpleGNN provides a simple way to run predefined and new custom GNNs on predefined 
and new custom benchmark datasets. The goal is to achieve a good comparison quality 
between different architectures.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api

Installation
------------

To install SimpleGNN, run:

.. code-block:: bash

    pip install -e .

Quick Start
-----------

Here's a simple example to get started:

.. code-block:: python

    from sample.core import GNNModel, create_model
    from sample.helpers import load_dataset, evaluate_model

    # Create a model
    model = create_model('base', input_dim=10, hidden_dim=64, output_dim=2)

    # Load a dataset
    dataset = load_dataset('example_dataset')

    # Evaluate the model
    metrics = evaluate_model(model, dataset)

API Reference
-------------

For detailed API documentation, see the API reference section.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
