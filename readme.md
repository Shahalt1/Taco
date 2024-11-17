# README

## Overview

This repository contains a Streamlit-based web application that leverages advanced AI tools and models for processing and interaction. Follow the steps below to set up and run the application.

---

## Prerequisites

- **Python**: Ensure Python (3.7 or higher) is installed on your system. [Download Python](https://www.python.org/downloads/)
- **Google API Key**: Obtain your Google API key to use in the application. This will be entered when running the app.

---

## Installation Steps

### 1. Install Python (if not installed)

- Check if Python is already installed:
  ```bash
  python --version
  ```
- If not, download and install Python from [here](https://www.python.org/downloads/).

### 2. (Optional) Install `venv` Module

- If you want to use a virtual environment, ensure the `venv` module is installed:
  ```bash
  pip install virtualenv
  ```

### 3. (Optional) Create and Activate Virtual Environment

- **Create the virtual environment:**
  ```bash
  python -m venv venv
  ```
- **Activate the virtual environment:**
  - On Windows:
    ```bash
    venv\Scripts\activate
    ```
  - On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

### 4. Install Necessary Modules

- Install the required Python libraries using `pip`:
  ```bash
  pip install streamlit google-generativeai chromadb sentence-transformers numpy langchain-text-splitters pypdf
  ```

### 5. Run the Application

- Start the Streamlit app by running the following command in the terminal:
  ```bash
  streamlit run app.py
  ```

---

## Using the Application

1. Launch the application using the steps above.
2. Once the web app opens, it will prompt you to insert your **Google API Key**.
   - Enter your key to enable Google Generative AI functionalities.

---

## Screenshot

Below is a screenshot of the application:

![App Screenshot](app.png)

## Support

If you encounter any issues, feel free to open an issue in this repository. Contributions and suggestions are always welcome!

---

### License

This project is licensed under the MIT License.
