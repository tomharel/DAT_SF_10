# ROC Curve

def plot_roc_curve(target_test, target_predicted_proba):
	from sklearn.metrics import roc_curve
	from sklearn.metrics import auc
	
    "Plots and saves an ROC curve"
	fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="best")
    plt.savefig('roc' + str(time.time()) + '.png',format='png',dpi=600)