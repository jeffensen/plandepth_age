# Planning depth in young and old adults
Investigating differences in planning depth between young and old adults using the [Space Adventure Task](https://github.com/dimarkov/sat) and
a value iteration model described in https://github.com/dimarkov/pybefit/tree/master/examples/plandepth.

Requirements
------------
    pybefit

Installation
------------
Clone the PyBefit library
```sh
git clone https://github.com/dimarkov/pybefit.git
cd pybefit
```
and follow the installation instruction in the [README](https://github.com/dimarkov/pybefit) file. One 
can install the package either using poetry or anaconda package menager. 

Usage
------------
Running the following scripts performs model based data analysis and generates relevant figures:

| Nr. | Script  | Description |
| ------------- | ------------- | ------------- |
| 1  |  fit_behavior.py  | ...performs the inference of model parameters from Space Adventure Task raw data.  |
| 2  |  create_analysis_datasets.py  | ...merges inference results and raw data of all tasks into .CSV files.  |
| 3  |  results_plot_allinOne.py  | ...generates main results plot from .CSV dataset.  |
| 4  |  data_analysis.sps  | ...performs all statistical analyses and generates regression plots.  |
