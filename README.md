Planetary Defense Decision Support System (DSS)

Project Overview

This repository contains a high-fidelity Planetary Defense Decision Support System (DSS) designed to identify Near-Earth Objects (NEOs) with potential hazard profiles. The system leverages a cost-sensitive machine learning pipeline to prioritize safety-critical detection, achieving a validated recall of 99.2% on NASA-sourced astronomical data.

Core Technical Contributions

Engineered Feature Set: Developed domain-specific features, including a Kinetic Energy Proxy ($v^2 \times \text{mass}$) and Size-to-Distance ratios, to capture the underlying physics of orbital impact risks.

Synthetic Minority Over-sampling Technique (SMOTE): Addressed extreme class imbalance in the NASA NeoWs dataset, synthesizing hazardous samples to train a robust decision boundary.

Log-Normalized Analytics: Implemented logarithmic feature scaling in the dashboard visualizations to ensure transparent interpretability of model drivers across disparate numerical magnitudes.

Microservices Orchestration: Architected the system using Docker Compose to decouple the FastAPI inference engine from the Streamlit analytical frontend, ensuring a scalable and portable deployment.

MLOps and Automation

Continuous Retraining: Integrated GitHub Actions to automate weekly model updates, fetching the latest orbital observations via the NASA API to mitigate data drift.

CI/CD Pipeline: Configured automated build testing to verify dependency integrity and container health upon every code commit.

System Architecture

The project follows a modular microservices structure:

backend/: A FastAPI REST API that performs real-time feature engineering and model inference.

frontend/: A Streamlit dashboard providing visual analytics, historical asteroid templates, and hazard assessments.

scripts/: Python modules for data ingestion and the SMOTE-XGBoost training pipeline.

models/: Versioned model binaries (.joblib) for reproducible research.

Deployment Instructions

1. Environment Setup

Clone the repository and create a .env file in the root directory containing your NASA API credentials:

NASA_API_KEY=your_api_key_here


2. Containerized Launch

Execute the following command to build and orchestrate the services:

docker-compose up --build


3. Access Points

Analytical Dashboard: http://localhost:8501

Inference API Documentation: http://localhost:8000/docs
