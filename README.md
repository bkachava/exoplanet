# Exoplanet Exploration - Machine Learning Assignment

## Background

The Kepler Space Observatory is a NASA-build satellite that was launched in 2009. The telescope
is dedicated to searching for exoplanets in star systems besides our own, with the  goal 
of possibly finding other habitable planets besides our own. 

The original mission ended in 2013 due to mechanical failures, but the telescope has
nevertheless been functional since 2014 on a "K2" extended mission.

Kepler had verified 1284 new exoplanets as of May 2016. As of October 2017, there are over 
3,000 confirmed exoplanets total (using all detection methods, including ground-based ones). 
The telescope is still active and continues to collect new data on its extended mission.

In this assignment, Machine Learning models, capable of classifying candidate exoplanets, 
were created.


## Assumptions

The data set used for the machine learning model is available at [Exoplanet Data Source](https://www.kaggle.com/nasa/kepler-exoplanet-search-results), but the csv located in the folder Notebook was used.

The algorithms used for the models were:

- Support Vector Machines (SVM) with Linear and Gaussian kernels. The SVM was suggested in the starter code, but 
   without kernel specification, so I tried two kernels.

- Random Forests. A second classifier was proposed, based on the results of the SVM.

Also, a hyper-parameter tuning with the *meta-estimator* function GridSearchCV() was used.
  

## Steps

### Preprocess the data

 * Clean the data and remove unnecessary columns. 
  Drop the columns rowid, kepid, kepoi_name, kepler_name, koi_pdisposition, koi_score, 
  koi_tce_delivname and those where all values were null.

 * Set the target as the column `koi_disposition` and get the target names.

 * Perform feature selection (40 in total)

 * Separate the data into training and testing data.
 
 * Use `MinMaxScaler` to scale the numerical data. 
   MinMaxScaler rescales the data set such that all feature values are in the range [0, 1].
   However, this scaling compresses all inliers in the narrow range [0, 0.005] 
   for the transformed number of data.
   MinMaxScaler is very sensitive to the presence of outliers.


### Tune Model Parameters

 * Train the models.
 
   __SVM__ is a model for classification and regression problems.  The SVM algorithm 
   finds the optimal hyperplane that separates data points with the largest margin possible. 
   It can solve linear and non-linear problems. 
 
 
   __Random forests__ are many decision trees, built on different random subsets (drawn 
   with replacement) of the data, and using different random subsets (drawn without 
   replacement) of the features for each split.  This makes the trees different from each other
   and makes them overfit to different aspects. Then, their predictions are averaged, 
   leading to a smoother estimate that overfits less.

 
 * Use `GridSearch` to tune model parameters.
 
   To increase the accuracy of the SVM classifiers, tuning parameters were explored: 1) the 
   regularization parameter termed as C parameter in Python’s Sklearn library), 
   that tells the SVM optimization how much you want to avoid misclassifying each training 
   example, and 2) the gamma parameter, that defines how far the influence 
   of a single training example reaches, with low values meaning ‘far’ and high values 
   meaning ‘close’. 
 
   For Random Forests the tuning parameters explored were: 1) max_features which is 
   the size of the random subsets of features to consider when splitting a node. 
   By setting max_features differently, you'll get a "true" random forest, and 
   2) max_depth that represents the depth of each tree in the forest. The deeper the 
   tree, the more splits it has, and it captures more information about the data. 

 * Get the results from the classifiers.

   After fitting the SVM models, the best scores for the training set and the best parameters
   were obtained, and after the predictions, the classification reports were printed.

   For the Random Forests, after fitting the model the __feature importances__ were printed 
   and the scores for the training and the testing data sets were calculated.
 

### Evaluate Model Performance

  __Results__

  `SVM Linear`
  
	 Training Data Score: 0.8533089356511131
	 Testing Data Score: 0.8444647758462946

	 Best Training Parameters: {'C': 50, 'gamma': 0.0001}
	 Best Training Score: 0.8827386398292162
 
	 Testing Scores
	                precision    recall  f1-score   support
	FALSE POSITIVE       0.84      0.66      0.74       523
	     CONFIRMED       0.75      0.86      0.80       594
	     CANDIDATE       0.98      1.00      0.99      1069

	     micro avg       0.88      0.88      0.88      2186
	     macro avg       0.86      0.84      0.84      2186
	  weighted avg       0.88      0.88      0.88      2186
  
  `SVM Radial Basis Function`
  
       Training Data Score: 0.8325709057639524
       Testing Data Score: 0.807868252516011
	 
       Best Training Parameters: {'C': 50, 'gamma': 0.005}
       Best Training Score: 0.8388228118328759

	Testing Scores
	               precision    recall  f1-score   support
    FALSE POSITIVE       0.70      0.54      0.61       523
	     CONFIRMED       0.67      0.78      0.72       594
	     CANDIDATE       0.98      1.00      0.99      1069

	     micro avg       0.83      0.83      0.83      2186
	     macro avg       0.78      0.77      0.77      2186
	  weighted avg       0.83      0.83      0.82      2186
  
  `Random Forests`
   
       Training Data Score: 0.9172003659652334
       Testing Data Score: 0.8924977127172918
	 
       Feature Importances - Top 5
	     koi_fpflag_co: 0.10882413723054071
	     koi_fpflag_nt: 0.09734814374013237
	     koi_fpflag_ss: 0.06554677640545675
	     koi_prad:      0.054709465771780555
	     koi_model_snr: 0.053357315667707524

     Feature description
	     koi_fpflag_co. Centroid Offset Flag. The source of the signal is from a nearby star, 
	       as inferred by measuring the centroid location of the image both in and out of transit, 
	       or by the strength of the transit signal in the target's outer (halo) pixels as 
	       compared to the transit signal from the pixels in the optimal (or core) aperture.
	  
	     koi_fpflag_nt. Not Transit-Like Flag. A Kepler Objects of Interest (KOI) whose 
	       light curve is not consistent with that of a transiting planet. This includes, 
	       but is not limited to, instrumental artifacts, non-eclipsing variable stars, 
	       and spurious detections.
		 
	      koi_fpflag_ss. Stellar Eclipse Flag. A KOI that is observed to have a significant 
		secondary event, transit shape, or out-of-eclipse variability, which indicates that 
		the transit-like event is most likely caused by an eclipsing binary. However, self luminous,
	        hot Jupiters with a visible secondary eclipse will also have this flag set, but with a 
	        disposition of PC.
		 
	      koi_model_snr. Transit Signal-to-Noise. Transit depth normalized by the mean 
	        uncertainty in the flux during the transits.
		 
	      koi_prad__. Planetary Radius (Earth radii). The radius of the planet. Planetary radius 
		is the product of the planet star radius ratio and the stellar radius.


  __Conclusion__

     Amongst the Linear kernel and the Gaussian kernel (Radial Basis Function), 
     the Linear kernel performed slightly better when predicting classes 
	 FALSE POSITIVE and CONFIRMED. 
	 
     Random Forests performed slightly better than Linear Kernel (89% vs. 88%) 
     when predicting classes, and the feature importances were obtained for 
     further research.


## Notebook

See the Jupyter Notebook [here](/Notebook/exoplanet.ipynb).

	 
## Resources

* [Scikit-Learn Tutorial Part 1](https://www.youtube.com/watch?v=4PXAztQtoTg)

* [Scikit-Learn Tutorial Part 2](https://www.youtube.com/watch?v=gK43gtGh49o&t=5858s)

* [Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)

* More on
  [SVM](https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989)
  
  [Machine Learning SVM](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)
  
  [Max Features Parameter Random Forests](https://stackoverflow.com/questions/23939750/understanding-max-features-parameter-in-randomforestregressor)
  
  [Tuning Random Forests](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d)
  
  [Explaining Feature Importance](https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e)
