# Decision Trees Algorithm
## Getting Started
The Decision Trees Algorithm is built with Python3 and rely on a few modules that can be found requirements.txt.

## Installing dependencies
### Linux
1. Download any version of Python greater than 3.6 (minimum version of 3.5.2 is required however 3.6.0 and above is recommended) from  [this link](https://www.python.org/downloads/) and follow the installation instructions.

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
dta.py [-p] [-v] [-t] [-o] [-s] <data.mat>
```

### Flags
* **-p** flag enables tree pruning which is used to minimise the impact of noisy data.

* **-o** flag attempts to optimise the tree when learning algorithm is performed (termination criteria becomes that the algorithm stops when majority of the targets have the same value).

* **-v** starts a Flask based light-weight web server at [http://localhost:5000](http://localhost:5000) to access trees at [/show/:id](localhost:5000/show/1).

* **-t** flag runs a k-fold cross validation test on the generated trees and returns the confusion matrix.

* **-n** flag normalises the values in the confusion matrix returned by the cross validation.

* **-s** saves the generated trees in .mat format in the out directory (the first tree is `1.mat` and so on).

Combined use of **-p** and **-o** is supported however not encouraged. Based on our evaluation the performance is the best when only one of them is turned out, otherwise there might be issues with over-pruning.

> Please note that **-t** is not compatible with **-s** and **-v**, and **-n** has no effect when **-t** is off.