############
Quantifying GHG emissions by work
############


What is it
==========


  

Where to get it
===============



Quickstart    
==========


To use it call

.. code:: python

    import pymrio
    test_mrio = pymrio.load_test()

The test mrio consists of six regions and eight sectors:  

.. code:: python


    print(test_mrio.get_sectors())
    print(test_mrio.get_regions())

The test mrio includes tables flow tables and some satellite accounts. 
To show these:

.. code:: python

    test_mrio.Z
    test_mrio.emissions.F
    
However, some tables necessary for calculating footprints (like test_mrio.A or test_mrio.emissions.S) are missing. pymrio automatically identifies which tables are missing and calculates them: 

.
Contributing
=============

Want to contribute? Great!
Please check `CONTRIBUTING.rst`_ if you want to help to improve Pymrio.
  
.. _CONTRIBUTING.rst: https://github.com/konstantinstadler/pymrio/blob/master/CONTRIBUTING.rst
   


