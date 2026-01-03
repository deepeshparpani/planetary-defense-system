Planetary Defense Decision Support System (DSS) by dannycodes :)

🚀 Live Production Environment

Analytical Dashboard: https://asteroid-frontend-617598390128.us-central1.run.app

Inference API Documentation: https://asteroid-backend-617598390128.us-central1.run.app/docs

Project Overview

This repository contains a high-fidelity Planetary Defense Decision Support System (DSS) designed to identify Near-Earth Objects (NEOs) with potential hazard profiles. The system leverages a cost-sensitive machine learning pipeline to prioritize safety-critical detection, achieving a validated recall of 99.2% on NASA-sourced astronomical data.

📈 Project Evolution & Methodology

The project followed an iterative research path to move from a basic proof-of-concept to a production-grade MLOps pipeline:

Initial Discovery (200 cases): The project began with a small pilot dataset of 200 NASA observations. Early testing revealed that traditional classifiers struggled with the extreme class imbalance (less than 10% of objects are hazardous), leading to high false-negative rates—a critical failure for planetary defense.

Data Scaling (10,000 cases): To improve generalization, the dataset was scaled to 10,000 cases via the NASA NeoWs API. This larger volume allowed for more robust statistical validation but intensified the "Needle in a Haystack" problem.

Pipeline Evolution:

The Problem: Accuracy was high, but Recall was low (the model missed dangerous asteroids).

The Solution: We implemented a multi-stage pipeline. First, we applied SMOTE (Synthetic Minority Over-sampling Technique) to the 10,000 cases to synthetically balance the hazard class. Second, we moved from simple Logistic Regression to XGBoost with a custom weighted loss function to penalize missing a hazardous object more heavily than a false alarm.

Final Architecture: This evolved into the current decoupled microservices architecture to ensure that the heavy computational logic of feature engineering doesn't interfere with the user-facing analytical dashboard.

📚 Data Dictionary & Feature Engineering

The model processes raw observations from the NASA NeoWs (Near Earth Object Web Service) API.

Feature

Type

Description

est_diameter_min

Float

Estimated minimum diameter of the object (km).

relative_velocity

Float

Velocity relative to Earth at the time of close approach (km/h).

miss_distance

Float

The distance by which the object missed Earth (km).

absolute_magnitude

Float

The intrinsic luminosity of the object (H); lower values indicate larger objects.

kinetic_proxy

Engineered

Calculated as $v^2 \times \text{mass\_estimate}$. Represents the potential impact energy.

size_dist_ratio

Engineered

Ratio of diameter to miss distance. Highlights small objects passing extremely close.

velocity_dist_ratio

Engineered

Ratio of velocity to miss distance. Captures the "angular speed" of the approach.

Core Technical Contributions

Engineered Feature Set: Developed domain-specific features to capture the underlying physics of orbital impact risks.

Synthetic Minority Over-sampling Technique (SMOTE): Addressed extreme class imbalance in the NASA dataset to train a robust decision boundary.

Log-Normalized Analytics: Implemented logarithmic feature scaling in dashboard visualizations to ensure transparent interpretability.

Microservices Orchestration: Architected the system using a decoupled microservices pattern on Google Cloud Run.

MLOps and Automation

Continuous Retraining (CT): Integrated GitHub Actions to automate weekly model updates via the NASA API.

GitOps CI/CD: Configured a pipeline that automatically builds images via Cloud Build and updates Cloud Run services on every push to main.

Automated Health Monitoring: Implemented a "Heartbeat" monitoring script (scripts/health_check.py) to validate the full end-to-end inference loop.

System Architecture

The project follows a modular cloud-native structure:

backend/: FastAPI REST API for real-time feature engineering and model inference.

frontend/: Streamlit dashboard providing visual analytics and hazard assessments.

scripts/: Python modules for SMOTE-XGBoost training and production health monitoring.

cloudbuild.yaml: Infrastructure-as-Code (IaC) defining build and deployment steps.

Local Development

To run a local instance for development purposes:

# Build and orchestrate the services
docker-compose up --build


Verification & Monitoring

The system health can be verified locally using the custom dashboard script:

python scripts/health_check.py

