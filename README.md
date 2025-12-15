# Simple Pendulum Project

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone
2. **Navigate to the Project Directory**
   ```bash
   cd simple-pendulum
3. **Create a Virtual Environment**
   ```bash
   python -m venv venv
4. **Activate the Virtual Environment**
5. 
   - On Windows:
     ```bash
     venv\Scripts\activate
     
    - On macOS/Linux:
   ```bash
     source venv/bin/activate

5. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```
   
## Running the Project
1. **Run MLFLOW with SQLite and Local Artifacts** 
   ```bash
    mlflow server --backend-store-uri sqlite:///scripts/mlflow.db --default-artifact-root ./mlartifacts
   ```
2. **Run any training script from `scripts` folder** 

3. **Run Jupyter Lab** 
    ```bash
    jupyter lab
   ```

