# README

## Overview

This repository contains a Streamlit-based web application that leverages advanced AI tools and models for processing and interaction. Follow the steps below to set up and run the application.

---

## Prerequisites

Ensure the following tools are installed on your system before proceeding:

1. **Python**: Download and install Python from [Python's official website](https://www.python.org/).
2. **Git**: Install Git by visiting [Git's official website](https://git-scm.com/).

---

## Installation Steps

### Step 1: Clone the Repository or Download as ZIP

- **Option 1 (Clone the Repo):**
  ```bash
  git clone https://github.com/Shahalt1/Taco.git
  ```
- **Option 2 (Download as ZIP):**
  - Go to [https://github.com/Shahalt1/Taco.git](https://github.com/Shahalt1/Taco.git).
  - Click on the green "Code" button and select "Download ZIP."
  - Extract the contents to a folder of your choice.

---

## Installation Steps

### 1. Install Python (if not installed)

- Open 'cmd' or 'PowerShell'

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
