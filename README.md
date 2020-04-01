# Project: Can you recognize the emotion from an image of a face? 
<img src="figs/CE.jpg" alt="Compound Emotions" width="500"/>
(Image source: https://www.pnas.org/content/111/15/E1454)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2020

+ Team 9
+ Team members
	+ Huang, Huize
	+ Wang, Mengchen
	+ Wang, Rui
	+ Wu, Jiadong
	+ Zhang, Qin 

+ Project summary: In this project, we created a classification engine for facial emotion recognition. For feature selection, we tried Principle Component Analysis(PCA) to reduce the dimension of the features but found out it didn't help for most models. After trying multiple machine learning models such as Gradient Boosting Machine (GBM), Linear discriminant analysis (LDA), Logistic Regression(LR), Support Vector Machine (SVM), Random Forest (RF), XGBoost (XGB), ensemble model of SVM and XGB, and Neural Network (NN), we found that using Neural Network model has better effect. The details of baseline model (GBM) and advanced model(NN) is shown as below.
  + Baseline model(GBM) achieved approximately 42% accuracy and need about 30min to train
  + Advanced model(NN) achieved approximately 56.2% accuracy and only need about 35s to train
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) (In the following context, "all team members" refers to the team meambers listed above.) All team members listed above discussed and decided the models to build in this project.
  For the model training, Qin Zhang designed and trained the baseline model (GBM) in R. Rui Wang conducted PCA on the data and designed and trained the SVM model in R. Mengchen Wang designed and trained LDA and Logistic Regression model in R. Jiadong Wu designed and trained Random Forest model in R and Neural Network model in Pytnon. Huize Huang designed and trained XGboost model in R and designed the ensemble model. The ensemble model is trained by Rui. 
  For the other work, Rui integrated the codes of main file, with Huize providing some edits. Qin and Mengchen worked on the presentation slides. Huize contributed to the readme files and contribution statement on github.
  Ryan Walters was assigned to our group at first. He joined the first online meting and offered to designed and trained the Convolutional Neural Network model. But he never share his work or code with the rest of the team and hasn't replied to the rest of the team since Mar.29th.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
