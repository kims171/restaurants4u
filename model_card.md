# Model Card: restaurants4u Restaurant Recommendation Classifier

## Model Overview

The restaurants4u model is a Random Forest classifier used to predict whether a restaurant is likely to be highly rated. The model supports a restaurant recommendation pipeline by helping rank or filter restaurants based on structured restaurant metadata.

## Model Type

- Model family: Random Forest Classifier
- Task type: Binary classification
- Target variable: `is_highly_rated`
- Positive class: Restaurant rating greater than or equal to 4.0

## Intended Use

The model is intended to support restaurant recommendation and ranking in the restaurants4u application. It can help identify restaurants that are more likely to be highly rated based on metadata such as location, category, and availability of contact details.

## Not Intended Use

The model should not be used for financial, legal, health, employment, or safety-critical decisions. It should not be treated as a complete measure of restaurant quality because it does not fully capture personal taste, recent service changes, dietary needs, accessibility, or live customer feedback.

## Dataset

The model is trained on restaurant metadata. The dataset includes information such as restaurant category, location, rating, website availability, phone availability, and image availability.

## Features

Example features include:

- Latitude
- Longitude
- Encoded restaurant category
- Website availability
- Phone availability
- Image availability

## Evaluation Metrics

The model should be evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Balanced accuracy

## Risks and Limitations

The model may produce stale recommendations if restaurant data changes over time. It may also over-represent popular categories or regions if the training data is imbalanced. The model does not fully capture individual user preferences, price preferences, dietary restrictions, accessibility needs, or recent service quality.

## Monitoring Plan

The production system should monitor:

- Prediction distribution
- Positive recommendation rate
- Category distribution drift
- Location distribution drift
- Rating distribution drift
- Missing value rate
- API latency
- API error rate

EvidentlyAI is used to generate drift reports by comparing reference data with recent production-like data.

## Retraining Plan

Retraining should be triggered when drift is detected for consecutive monitoring periods, when model performance drops below the accepted baseline, or when new restaurant data significantly differs from the training distribution.

A candidate retrained model should only be promoted if it improves or maintains key metrics such as F1 score, recall, and balanced accuracy.

## Ethical Considerations

The model should be monitored for geographic and category bias. Recommendation systems can unintentionally over-promote restaurants from already popular neighborhoods or categories. Future versions should include fairness checks, user feedback, A/B testing, and better personalization controls.