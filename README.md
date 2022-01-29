# Pegasos algorithm for one-class SVM (one_class_svm_tool)

## Introduction

"Pegasos algorithm for one-class SVM" is a training algorithm for one-class support vector machinie (one-class SVM). This algorithm is much faster than the standard one-class SVM without loss of performance in the case of linear kernel. 

### References

* Changki Lee. Pegasos Algorithm for One-Class Support Vector Machine. IEICE Transactions on Information and Systems, Vol.E96-D, No.5, May 2013, pp.1223-1226.
* Changki Lee. 1-Slack One-Class SVM for Fast Learning. Journal of KIISE, ISSN:1229-7712, VOL.19, NO.5, May 2013, pp.253-257.

## Data format

This is the training/testing file format of one_class_svm_tool.

### binary feature
    [label] [string_feature1] [string_feature2] ...
    [label] [string_feature1] [string_feature2] ...

label is the class of your one-class SVM (1 or -1).
string_feature_n is a feature string that have 1 as a feature value.

### general feature
    [label] [string_feature1:value] [string_feature2:value] ...
    [label] [string_feature1:value] [string_feature2:value] ...

## Usage

A command line utility to train/test one-class SVM.

### Usage: one_class_svm_tool [OPTIONS]... [FILES]...
    -h, --help             Print help and exit
    -V, --version          Print version and exit

### Training options:
    -c, --cost=FLOAT       set Cost of one class SVM (cost=1/lambda for SGD) (default=`1')                    
    -m, --model=STRING     set model file name
        --skip_eval        skip test set evaluation in the middle of training (default=off)
    -r, --random=INT       use random_shuffle in train_data (disabled if use 0) (default=`0')
        --train_num=INT    set number of sentence in train_data for training (for experiments) (disabled if use 0)  (default=`0')
    -i, --iter=INT         iterations for training algorithm (sgd) (default=`100')
    -t, --tol=FLOAT        set Tolerance  (default=`1e-03')
        --period=INT       save model periodically (sgd)  (default=`0')
    -v, --verbose          verbose mode  (default=off)

### Predict options:
    -o, --output=STRING    prediction output filename

### one class SVM options:
    -e, --epsilon=FLOAT    set epsilon  (default=`0.001')
        --buf=INT          number of new constraints to accumulated before recomputing the QP (sgd)  (default=`10')
        --rm_inactive=INT  inactive constraints are removed (iteration) (one_slack_smo)  (default=`50')

### Kernel options:
    --kernel=INT           kernel type (0=Linear 1=POLY 2=RBF 3=SIGMOID in SVMs) (default=`0')
    --gamma=FLOAT          gamma in RBF kernel (in SVMs)  (default=`1e-5')
    --coef=FLOAT           coef in POLY/SIGMOID kernel (in SVMs)  (default=`1')
    --degree=INT           degree in POLY kernel (in SVMs)  (default=`3')

 ### Group: MODE
    -p, --predict          prediction mode, default is training mode
        --show             show-feature mode
        --remove_zero      remove zero-value features (with --tol threshold)

 ### Group: Parameter Estimate Method
    --one_slack            use 1-slack one class SVM without Gram matrix (linear kernel only)
    --one_slack2           use 1-slack one class SVM using Gram matrix (linear kernel only)
    --sgd                  use SGD (Pegasos) in primal optimization (random shuffled train_data)
    --sgd2                 use SGD (Pegasos) in primal optimization (pick random examples)
                           
### Example of options:
    $ one_class_svm_tool --sgd -c 1000 -m model.txt train.feature.txt test.feature.txt
    $ one_class_svm_tool -p -m model.txt -o output.txt test.feature.txt
