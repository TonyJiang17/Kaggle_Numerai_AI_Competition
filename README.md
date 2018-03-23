# Kaggle Numerai 2017 Tournament 52

I built a model for the datasets using two different classifiers from scikit-learn. Read the blog post [here](http://techinpink.com/2016/09/21/numerai-artificial-intelligence-tournament/.)

Competition Content (From Kaggle)
You will find 21 features in the datasets. All of them range between 0 and 1. These are encrypted features, that later (after your prediction submission) will be de-crypted back to trade-able signals.

Target Feature is a Boolean ( 0 or 1 ).

Your Prediction CSV File has to have two columns: id , prediction

The prediction column , unlike the target feature, has to range between 0 and 1 (float,double) assigning the probability of row[id] being a 1.

Acknowledgements
This Dataset was downloaded from www.numer.ai

I built a model for the datasets using two different classifiers from scikit-learn. 

```datasets folder contains the training sets and test sets```

``` scripts folder contains the two different classifiers explored in making predictions to these datasets```

```Predictions made are shown by CSV : prediction_id and prediction_prob. The probability column is the probability estimated by the model of the observation being 1.```
