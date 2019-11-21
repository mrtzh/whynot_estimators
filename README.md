# WhyNot Estimators 

A collection of causal inference estimators implemented in Python and R to pair
with the Python causal inference library [whynot](https://github.com/zykls/whynot). 
For more information, check out the [documentation](https://whynot-docs.readthedocs-hosted.com/en/latest/).

## Installation
You can perform a minimal installation of `whynot_estimators` with
```
git clone https://github.com/zykls/whynot_estimators.git
cd whynot_estimators
pip install -r requirements.txt
```
You can also install via pip
```
pip install whynot_estimators
```

This installs the basic framework. Additional estimators, along with their
dependencies are installed separately. To see a list of all available
estimators, use
```
python -m whynot_estimators show_all
```
To install a particular estimator, e.g. the `causal_forest`, run
```
python -m whynot_estimators install causal_forest
```
Note this estimator requires a working `R` installation. The `show_all` command
also shows which estimators require `R`.

To install all of the estimators, use
```
python -m whynot_estimators install all
```
Alternatively, you can install the dependencies for a specific estimator by hand
by looking [here](whynot_estimators/__main__.py).


## Installing R
Some estimators in `whynot_estimators` require a functioning R installation. One
way to satisfy this requirement is using conda. 
```
# Install Anaconda (Linux)
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
bash Anaconda3-2018.12-Linux-x86_64.sh

# Install Anaconda (MacOSx)
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-MacOSX-x86_64.sh
bash Anaconda3-2019.03-MacOSX-x86_64.sh

# Create R environment
conda create --name whynot r-essentials r-base
```
