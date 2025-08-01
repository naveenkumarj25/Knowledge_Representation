# `KNOWLEDGE REPRESENTATION`

Deployed Web URL - https://knowledgerepresentation.streamlit.app

## Table of Contents
1. [Objective](#objective)
2. [Problem Description](#problem-description)
3. [About the Project](#about-the-project)
4. [How It Works](#how-it-works)
    -   [Process Flow](#process-flow)
5. [Technical Details](#technical-details)
    - [Architecture](#architecture-diagram)
    - [Key Technologies and Libraries](#key-technologies-and-libraries)
6. [Datasets Utilized and Analyzed](#Datasets-Utilized-and-Analayzed)
   - [Weather Pattern Analysis(Classification)](#weather-pattern-analysis-(classification))
   - [Real Estate Market Insights(Regression)](#real-estate-market-insights-(regression))
7. [How to run the project](#how-to-run-the-project)

---
## Objective: 
The primary objective of this project is to develop an AI-based solution that can effectively represent 
knowledge and generate insights from any structured dataset. The solution should be capable of 
processing and analyzing structured data, identifying patterns, and generating meaningful insights that 
can aid in decision-making processes.
---
## Problem Description: 
In the era of big data, organizations across various sectors are generating massive amounts of data 
every day. This data, if processed and analyzed correctly, can provide valuable insights that can 
significantly improve the decision-making process. However, the challenge lies in effectively 
representing this knowledge and extracting useful insights from it.

Your task is to develop an AI-based solution that can handle this challenge. You will be provided with a 
structured dataset. Your solution should be able to process this dataset, represent the knowledge 
contained within it effectively, and generate meaningful insights.

The solution should include the following features:

1. **Data Pre-processing**: The solution should be able to clean and pre-process the dataset to make it 
suitable for further analysis.
2. **Knowledge Representation**: The solution should effectively represent the knowledge contained 
within the dataset. This could be in the form of graphs, charts, or any other visual representation 
that makes the data easy to understand.
3. **Pattern Identification**: The solution should be able to identify patterns within the dataset. This 
could include identifying trends, anomalies, or any other patterns that could provide valuable 
insights.
4. **Insight Generation**: Based on the identified patterns, the solution should generate meaningful 
insights. These insights should be presented in a clear and understandable manner.
5. **Scalability**: The solution should be scalable. It should be able to handle datasets of varying sizes 
and complexities.
6. **User-friendly Interface**: The solution should have a user-friendly interface that allows users to 
easily interact with it and understand the generated insights
---
## About the Project

Knowledge Representation is an advanced AI solution that transforms raw data into actionable knowledge. By leveraging machine learning and natural language processing, our tool provides a user-friendly interface for data analysis, visualization, and insight generation. It addresses the critical need for efficient data processing and insight extraction in the age of information overload.

---
## How It Works

### Process Flow

<div align="center">
    <img src = "reports/img/Knowrep Process diagram.png" title = "Process Flow" height="650px" >
</div>

1. **Upload**: Users upload their CSV files through the Streamlit interface.
2. **Process**: The application pre-processes the data, handling encoding detection and basic cleaning.
3. **Analyze**: Leveraging machine learning algorithms, the tool identifies patterns and generates insights.
4. **Visualize**: Results are presented through clear, interactive visualizations using Matplotlib and Seaborn.
5. **Interact**: Users can ask questions about their data using natural language, which are answered by our AI agent. The User questions are converted into `SQL` queries and executed on the dataset. The results are displayed in the understandable format.
6. **Predict**: For applicable datasets, users can run machine learning predictions using various algorithms.
---
## Technical Details

## Architecture Diagram:
![Architecture Diagram](<reports/img/Knowrep Architecture diagram.png>)

### Key Technologies and Libraries

- **Python**: The core programming language used.
- **Streamlit**: For building the web-based user interface.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning algorithms and data preprocessing.
- **XGBoost**: For gradient boosting machine learning.
- **Matplotlib & Seaborn**: For data visualization.
- **LangChain**: For building applications with large language models.
- **SQLAlchemy**: For database operations and SQL query generation.
- **SQLite**: For storing and querying data.
- **Google Generative AI LLM**: For natural language processing and generation, we used LLM: `gemini-pro`
---
## Example
## Datasets Utilized and Analayzed

### Weather Pattern Analysis (Classification)
[Weather Dataset](datasets/weather_classification_data.csv)

#### Dataset Overview
11,586 weather observations with 11 features including temperature, humidity, wind speed, and atmospheric conditions.

#### Key Findings
1. **Temperature-Humidity Correlation**: Strong positive correlation (0.71) indicating a significant relationship between temperature and humidity levels.
2. **Precipitation Dynamics**: Moderate positive correlation (0.42) between wind speed and precipitation probability, suggesting increased rainfall likelihood during windy conditions.
3. **Visibility Factors**: Moderate negative correlation (-0.40) between cloud cover and visibility, highlighting the impact of cloud density on visual range.

#### Actionable Insights
1. Implement real-time monitoring systems for temperature and humidity to mitigate health risks during extreme weather events.
2. Develop predictive models for precipitation based on wind speed data to improve weather forecasting accuracy.
3. Optimize outdoor activity planning and travel logistics based on visibility and cloud cover predictions.
---
### Real Estate Market Insights (Regression)

[MELB Dataset](datasets/melb_data.csv)

#### Dataset Overview
Comprehensive data on Melbourne's real estate market, including property characteristics, pricing, and geographical information.

#### Key Findings
1. **Price Distribution**: Right-skewed distribution with a median of $870,000, indicating a concentration of properties in the mid-range market.
2. **Geographical Trends**: Premium properties clustered in suburbs like Brighton and Toorak, with more affordable options in areas such as Werribee and Melton.
3. **Property Type Analysis**: Houses dominate the market (70% of listings) and command higher prices compared to units and townhouses.

#### Actionable Insights
1. Develop targeted marketing strategies for different suburbs based on property types and local price points.
2. Optimize pricing models to account for seasonal fluctuations and preferred sale methods in different areas.
3. Focus urban development initiatives on high-value areas to enhance property desirability and values.
---
## How to run the project:

1. Clone the repository

2. Create a virtual environment (ie., conda) with Python >= 3.8 and activate it.

3. Install the required libraries using the following command:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the following command to start the application:

    ```bash
    streamlit run Main.py
    ```

5. The application will open in the default browser and you can start using it.
<div align="center">
    <img src = "reports/img/UI_screenshot.png" height = 80% width = 80% title = "Main Page" >
</div>

6. You need to get your free gemini API key from [here](https://aistudio.google.com/app/apikey) and enter it in the app's sidebar and then upload the dataset (ie., .csv file).

7. You can generate insights, Chat with CSV data, and predict the data using the app.

8. Make sure to `Reset Application` and refresh the page before uploading a new dataset.

9. You can even try out our deployed application.

10. Any queries or suggestions can be raised as an issue in the repository.
