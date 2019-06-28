Supported Dataset Types
====================================================================

Dataset URIs must have the protocols of either ``http`` or ``https``.

.. note::
    
    You can alternatively use relative (e.g. ``data/dataset.zip``) filepaths as dataset URIs, 
    only if you have deployed the full Rafiki stack on your own machine. This filepath is relative to
    the root of the project directory.

.. note::

    Refer to `./examples/datasets/ <https://github.com/nginyc/rafiki/tree/master/examples/datasets/>`_ for examples on pre-processing 
    common dataset formats to conform to the Rafiki's own dataset formats.


.. _`dataset-type:IMAGE_FILES`:

IMAGE_FILES
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``images.csv`` at the root of the directory.

The ``images.csv`` should be of a `.CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`_
format with 2 columns of ``path`` and ``class``.

For each row,

    ``path`` should be a file path to a ``.png``, ``.jpg`` or ``.jpeg`` image file within the archive, relative to the root of the directory.

    ``class`` should be an integer from ``0`` to ``k - 1``, where ``k`` is the number of classes in the classification of images.

An example of ``images.csv`` follows:

.. code-block:: text

    path,class
    image-0-of-class-0.png,0
    image-1-of-class-0.png,0
    ...
    image-0-of-class-1.png,1
    ...
    image-99-of-class-9.png,9
    

.. _`dataset-type:CORPUS`:

CORPUS
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``corpus.tsv`` at the root of the directory.

The ``corpus.tsv`` should be of a `.TSV <https://en.wikipedia.org/wiki/Tab-separated_values>`_ 
format with columns of ``token`` and ``N`` other variable column names (*tag columns*).

For each row,

    ``token`` should be a string, a token (e.g. word) in the corpus. 
    These tokens should appear in the order as it is in the text of the corpus.
    To delimit sentences, ``token`` can be take the value of ``\n``.

    The other ``N`` columns should be integers from ``0`` to ``k_i - 1``, where ``k_i`` is the number of classes for each column.
    These tag columns describe the corresponding token as part of the text of the corpus, and depends on the task.


An example of ``corpus.tsv`` for POS tagging follows:

.. code-block:: text

    token       tag
    Two         3
    leading     2
    ...
    line-item   1
    veto        5
    .           4
    \n          0
    Professors  6
    Philip      6
    ...
    previous    1
    presidents  8   
    .           4
    \n          0
