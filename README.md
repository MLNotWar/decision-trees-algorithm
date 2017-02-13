# Decision Trees Algorithm
## Getting Started
The Decision Trees Algorithm is built with Python3 and rely on a few modules that can be found requirements.txt.

## Installing dependencies
### Linux
1. Download any version of Python greater than 3.6 from  [this link](https://www.python.org/downloads/) and follow the installation instructions.

2. Install pip with the following command:
```
curl https://bootstrap.pypa.io/get-pip.py | python -
```

3. Navigate to the project repository and install requirements with the following command:
```
pip install -r requirements.txt
```

> Note: Alternatively you can run Python in a virtual environment such as virtualenv or pyenv.

## Running the Application
Now that your environment is ready you can start building trees with the following command:

```
dta.py [-p] [-v] [-t] [-o] <data.mat>
```

### Flags
* **-p** flag enables tree pruning which is used to minimise the impact of noisy data.

* **-v** starts a Flask based light-weight web server at [http://localhost:5000](http://localhost:5000) to access trees at [/show/:id](localhost:5000/show/1).

* **-t** flag runs a k-fold cross validation test on the generated trees and returns the confusion matrix.

* **-o** saves the generated trees in .mat format in the out directory.
