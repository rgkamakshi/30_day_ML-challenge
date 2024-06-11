**Solar Home Electricity Forecasting**

**Contributors**
    Gnanambal Kamakshi Renganathan (Myself)
    
    Thrushwanth Kakuturu
    
    Sireesha Pulipati(Team Lead)

**Overview**

Ausgrid is a major electricity distributor in Australia, responsible for managing and distributing electricity to residential, commercial, and industrial consumers. Recently, Ausgrid has focused on integrating renewable energy sources into its grid infrastructure to reduce carbon emissions and promote sustainability.

Solar electricity generation is a key component of this renewable energy transition. However, photovoltaic energy is volatile as it depends on weather conditions, making it challenging for network operators to integrate and manage. Renewable energy forecasting is a crucial solution to effectively manage this volatility within the power system.

This project aims to develop a robust forecasting model to predict energy consumption, ensuring optimal energy management and proper control of energy supplied to users.
Data Source

The data for this project is sourced from Ausgrid's Solar Home Electricity Data, available at Ausgrid Data to Share.
Datasets

**Monthly Data**: This dataset includes information from 2,657 solar homes and 4,064 non-solar homes, providing a comprehensive view of energy consumption patterns. It covers a period of 8 years (2007-2014), allowing for the identification of seasonal trends.
    Total Household Energy Needs: This metric accounts for the total electricity used by households, considering both solar generation and grid consumption. It can be estimated by: [ \text{Net Household Consumption} = \text{Total Grid Consumption} - \text{Solar Power Output} ]

**Forecasting Models**
SARIMA

    Model: SARIMAX(0, 1, 2)x(1, 0, 1, 12)
    Best Parameters from Grid Search: (0, 0, 2)
        Non-Seasonal Parameters: p=5, d=1, q=2
        Seasonal Parameters: P=0, D=0, Q=2 (seasonal MA term at lag 12)
    Evaluation Metrics:
        MAPE: 27.672
        RMSE: 149027.037
        Best Grid Search MAPE: 16.544
        Best Grid Search RMSE: 187509.107

XGBoost - For next Month Prediction

    Model: XGBRegressor
        Learning Rate: 0.05
        Max Depth: 5
        Subsample: 1.0
        Colsample by Tree: 0.8
        N Estimators: 300
    Evaluation Metrics:
        RMSE: 101.36
        MAPE: 40.54
        sMAPE: 25.24

**Development and Deployment**
Development

    Environment: Vertex AI Workbench
    Models Developed: XGBoost

Deployment (XGBoost)

    Generate Model Artifacts: Train the XGBoost model and save it as a model artifact.
    Upload to Google Cloud Storage (GCS): Upload the model artifact to a designated GCS bucket.
    Import the Model into Vertex AI: Import the model into Vertex AI for deployment.
    Deploy to Vertex Endpoint: Deploy the model to a Vertex endpoint for serving next month predictions for every customer.

**Technologies Used**

    Programming Languages: Python
    Libraries: XGBoost, Pandas, NumPy, Scikit-learn, Matplotlib, statsmodels
    Platform: Google Cloud (Vertex AI, Google Cloud Storage)

**Conclusion**
This project successfully developed and deployed an XGBoost model to forecast solar home electricity consumption. 
By encapsulating our XGBoost model into a container and creating a custom model using Google Cloud's Vertex AI, we have made it accessible for inference and predictions. 
This deployment provides a robust solution for managing renewable energy within the power system. 
Our forecasting model aids in optimal energy management, ensuring efficient control of energy supply and significantly contributing to Ausgrid's sustainability goals. 
Through this project, we have demonstrated how advanced machine learning techniques and cloud technologies can be leveraged to address critical challenges in renewable energy integration and management.
