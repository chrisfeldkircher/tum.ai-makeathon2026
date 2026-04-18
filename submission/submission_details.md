osapiens Terra
×
TUM.ai
Challenges
Leaderboard
Honey Badgers
Logout
Challenge 1
osapiens Challenge: Detecting Deforestation from Space
Build a multimodal ML system that detects deforestation events from satellite data, handles noisy supervision, and generalizes across geographic regions.

Why It Matters
Under EUDR, companies need scalable geospatial evidence that their supply chains are deforestation-free. Coffee and other agricultural supply chains are especially hard to verify because batches are repeatedly mixed, split, and recombined across many intermediaries.

The Challenge
You work with multimodal satellite data such as radar, optical imagery, foundation model embeddings, and weak supervision. The hard part is building a system that stays reliable despite noisy labels, inconsistent imagery, and geographic shift.

Your Mission
•
Detect deforestation events at pixel level from multimodal satellite time series.
•
Combine Sentinel-1, Sentinel-2, and embedding-based signals effectively.
•
Generalize to unseen regions instead of overfitting to one geography.
What We Evaluate
The live leaderboard in this app gives quantitative feedback during the hackathon, but the final outcome can still include jury-based qualitative evaluation.

•
Model design, reasoning, and use of data
•
Handling of noisy labels and imagery
•
Generalization across regions
•
Quantitative performance and scalability
•
Clarity, interpretability, and presentation quality
Bonus Ideas
•
Predict when deforestation occurred, at month or year level
•
Estimate confidence or explicitly model label uncertainty
•
Build a lightweight visualization or monitoring tool
Deadline: 19.04.2026, 11:00
Countdown: 15h 49m 43s(Europe/Berlin)
Metric: Union IoU
File format: GeoJSON (.geojson)
Required fields: geometry, properties.time_step (optional)
Scoring Metrics
Union IoU — Primary ranking metric. We compare the union of all predicted polygons against the union of all scored ground-truth polygons.
Polygon Recall — Overlap area between the prediction union and the spatial ground-truth union, divided by the spatial ground-truth area.
Polygon Level FPR — Predicted area outside the spatial ground-truth union, divided by total predicted area.
Year Accuracy — Correctly dated overlap area divided by the union area of predictions and the temporal ground-truth subset. Wrong years, missing time predictions, missing detections, and extra detections are penalized by area.
Submission usage

0 of 10 submission slots used. Failed submissions do not count.

10 remaining
Submit Solution
Upload a GeoJSON file (.geojson) with these required fields:

geometry
properties.time_step (optional)
Submit a GeoJSON FeatureCollection. Every feature should be a Polygon or MultiPolygon and can includeproperties.time_step in YYMM format, for example 2204 for April 2022.

Validation before upload

• file extension must be .geojson
• GeoJSON must parse and use a top-level FeatureCollection
• only Polygon and MultiPolygon are accepted
• time_step may be valid YYMM, null, or omitted
Limit: maximum 10 counted submissions per team for this challenge. Failed submissions do not count and do not trigger the 5-minute cooldown.

Download example GeoJSON
Drag & drop your .geojson file here, or

browse to upload
GeoJSON (.geojson) · Max 400 MB

Submit
