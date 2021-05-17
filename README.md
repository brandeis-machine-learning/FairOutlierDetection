# Deep Clustering for Fair Outlier Detection
Source code for kdd2021 research paper submission

## Paper Abstract
In this paper, we focus on the fairness issues regarding unsupervised outlier detection. Traditional algorithms, without specific design for algorithmic fairness, could implicitly encode and propagate statistical bias in data and raise societal concerns. To correct such unfairness and deliver a fair set of potential outlier candidates, we propose Deep Clustering-based Fair Outlier Detection (DCFOD) that learns a good representation for utility maximization while enforcing the learnable representation to be subgroup-invariant on the sensitive attribute. Considering the coupled and reciprocal nature between clustering and outlier detection, we leverage deep clustering to discover the intrinsic cluster structure and out-of-structure instances. Meanwhile, an adversarial training erases the sensitive pattern for instances for fairness adaptation. Technically, we propose an instance-level weighted representation learning strategy to enhance the joint deep clustering and outlier detection, where the dynamic weight module re-emphasizes contributions of likely-inliers while mitigating the negative impact from outliers. Demonstrated by experiments on eight datasets comparing to 17 outlier detection algorithms, our DCFOD method consistently achieves superior performance on both the outlier detection validity and two types of fairness notions in outlier detection.

## Software Requirement
```
numpy==1.7.1
torch==1.7.0
sklearn==0.22.0
pandas==1.0.5
cuda=10.1.243
```

Model training requires at least one GPU.

## Datasets
We compare performance on eight datasets [student](https://archive.ics.uci.edu/ml/datasets/student%2Bperformance), [asd](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult), [obesity](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+), [cc](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients), [german](http://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29), [drug](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29), [adult](https://archive.ics.uci.edu/ml/datasets/adult), [kdd](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29).

To prepare datasets before model training, run
```
python3 getDatasets.py
```
Theoretically, you don't have to modify any path for datasets or folders throughout model training. 

## DCFOD 
To obtain DCFOD's performance on a speicifc dataset, run
```
python3 train.py *dataset_name* *GPU_index* *with_weight*

i.e., 

python3 train.py student 0 true
```
`GPU_index` indicates the *i-1* th GPU you want to train on. If you only have one, simply type *0*.

To obtain the results for derivative **DCOD**, change the `kf` hyperparameter value to 0 in the *train* method.

### Competitive Method: FairLOF
FairLOF requires the baseline result of LOF, you should first run
```
python3 LOF.py *dataset_name*
```
followed by
```
python3 get_Ws_for_FairLOF.py
```
which will retrieve all the `Ws` variable based on LOF results for all datasets, which is requried in FairLOF calculation.

If you don't have the LOF results for all datasets, tweak the *datasets* list in the `get_Ws_for_FairLOF.py` file.

then run
```
python3 FairLOF.py *dataset_name*
```
the experiment runs on 4 GPUs, you can tweak line 21-25 in `FairLOF.py` to modify cuda and the associated GPU settings.

### Competitive Method: FairOD
We run FairOD with
```
python3 FairOD.py *dataset_name* *GPU_index* *fair_command*
```
you should first obtain a baseline result, i.e., 
```
python3 FairOD.py student 0 f
```
then train the fair model
```
python3 FairOD.py student 0 t 
```
### Competitive Method: Conventional Outlier Detectors
To save methods' outlier scores and obtain metrics' values on a specific dataset, run
```
python3 pyod_results.py *dataset_name*
```
Or, if you already ran the above command, which has saved outlier scores, and you want to re-obtain the metrics' values, simply run
```
python3 Retriever.py *dataset_name*
```

### Numerical Evaluation
We use *AUC* to measure detection validity, and *Fgap*, *Frank* to measure two types of fairness degree. 

We calculate *AUC* with `roc_auc_score` from `sklearn.metrics`, and define *Fgap*, *Frank* in `Retriever.py`.

During model training, we obtain the fairness metrics with the *fetch* method in `Retriever.py`.
