# Cross Validation with Flexible Threshold

def cross_validate(X, y, classifier, k_fold, min_p = 0.5) :
	"This function returns cross validation scores with flexible threshold"
    # derive a set of (random) training and testing indices
	k_fold_indices = KFold(len(X), n_folds=k_fold,shuffle=True, random_state=1)

	k_score_total = 0
	# for each training and testing slices run the classifier, and score the results
	for train_slice, test_slice in k_fold_indices :

		model = classifier(X[ train_slice  ], y[ train_slice  ])
        
        
		# Classify based on threshold
		prob_vector = model.predict_proba(X[ test_slice ])[:,1]
		class_vector = []
		for i in xrange(0,len(test_slice)):
			if prob_vector[i] >= min_p:
				class_vector.append(1)
			else:
				class_vector.append(0)
           
		# count good predictions
		good_pred_count = 0
		for j in xrange(0,len(test_slice)):
			if class_vector[j] == y[ test_slice ][j]:
				good_pred_count +=1
        
		k_score = float(good_pred_count)/float(len(test_slice))
		#k_score = model.score(X[ test_slice ], y[ test_slice ])

		k_score_total += k_score

    # return the average accuracy
	return k_score_total/k_fold