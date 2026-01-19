# Caravan Insurance Customer Prediction

## Problem Description

This project addresses the challenge of identifying potential customers for caravan (mobile home) insurance policies. The goal is to build a predictive model that can help an insurance company target marketing efforts more effectively by predicting which customers are most likely to purchase caravan insurance.

### Business Context

Insurance companies spend significant resources on marketing campaigns, but only a small fraction of customers actually purchase caravan insurance policies. The dataset reveals a highly imbalanced classification problem where only **~6% of customers** own a caravan insurance policy (CARAVAN = 1). This imbalance makes the prediction task challenging and requires careful model selection and evaluation metrics.

### How the Solution Will Be Used

The deployed prediction service accepts customer demographic and insurance policy data via a REST API and returns:
- A probability score indicating the likelihood of the customer purchasing caravan insurance
- A binary classification (potential customer: yes/no)

This enables the insurance company to:
- Prioritize marketing outreach to high-probability customers
- Optimize marketing budget allocation
- Improve conversion rates for caravan insurance products

### Dataset

The data comes from the **UCI Machine Learning Repository - Insurance Company Benchmark (COIL 2000)** dataset:
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000)
- **Records**: 9,822 customers
- **Features**: 85 features covering:
  - **Demographic data** (43 features): Customer subtype, age, household composition, religion, marital status, education, occupation, social class, home ownership, car ownership, income, purchasing power
  - **Insurance policy contributions** (21 features): Premium contributions across different insurance product types
  - **Number of policies** (21 features): Count of policies held per insurance product type
- **Target variable**: `CARAVAN` - Whether the customer owns a caravan insurance policy (binary: 0/1)

---

## Exploratory Data Analysis (EDA)

Extensive EDA was performed in `notebook.ipynb` covering:

### Data Quality Assessment
- **Missing Values**: No missing values found in the dataset (Total NAs = 0)
- **Data Types**: 1 categorical column (`ORIGIN`), 86 integer columns
- **Value Ranges**: All features examined via `df.describe()` showing count, mean, std, min, quartiles, and max values

### Target Variable Analysis
- **Class Distribution**: Highly imbalanced with ~94% non-buyers and ~6% caravan insurance buyers
- **Stratified Splitting**: Train/validation/test splits were stratified to maintain class proportions:
  - Training set: 60%
  - Validation set: 20%
  - Test set: 20%

### Feature Analysis

#### Demographic Features
- Analyzed distributions of customer subtypes (`MOSTYPE` - 41 categories)
- Examined age distribution (`MGEMLEEF`), household size (`MAANTHUI`)
- Investigated religion, education, occupation, and social class distributions
- Analyzed income brackets and purchasing power classes

#### Insurance Policy Features
- **Contribution Features (P* columns)**: Distribution analysis of premium contributions across 21 insurance product types
- **Policy Count Features (A* columns)**: Distribution analysis of number of policies held across 21 product categories
- Created histograms for all policy number features to visualize distributions

#### Correlation Analysis
- Computed **Spearman correlation coefficients** between features
- Generated correlation heatmaps for:
  - Contribution features (P* columns)
  - Policy count features (A* columns)
- Key correlations identified:
  - `PWALAND` & `PTRACTOR`: 0.567 (agriculture third-party insurance & tractor policies)
  - `PWAPART` & `PBRAND`: 0.482 (private third-party insurance & fire policies)
  - `AWALAND` & `ATRACTOR`: 0.522 (similar pattern in policy counts)
  - `AWAPART` & `ABRAND`: 0.516 (private third-party & fire policy counts)

---

## Model Training

Multiple models were trained and evaluated, progressing from simple baselines to tuned ensemble methods.

### Models Trained

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Dummy Classifier | 0.500 | Baseline (most frequent class) |
| Logistic Regression (L2) | ~0.50 | Poor performance, no tuning warranted |
| Decision Tree | ~0.65-0.70 | Significant improvement over baseline |
| Random Forest | ~0.70-0.75 | Better generalization |
| **XGBoost** | **0.693** | Best performing model (test set) |

### Hyperparameter Tuning

#### Decision Tree Tuning
- **Parameters tuned**: `max_depth` [3, 4, 5, 6, 7, 8, 9, 10], `min_samples_leaf` [1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30]
- Optimized to balance between ROC-AUC and overfitting prevention

#### Random Forest Tuning
- **Parameters tuned**: 
  - `n_estimators`: [50, 100, 150, 200, 250, 300]
  - `max_depth`: [3, 4, 5, 6, 7, 8]
- Best configuration: `n_estimators=200+`, `max_depth=5-7`
- Used entropy criterion for information gain

#### XGBoost Tuning (Final Model)
Systematic parameter tuning with validation monitoring:

- **Learning rate (`eta`)**: Tested [0.05, 0.1, 0.2, 0.3]
  - Selected: **0.05** (slower learning, better generalization)
  
- **Tree depth (`max_depth`)**: Tested [3, 4, 5, 6]
  - Selected: **3** (prevents overfitting)
  
- **Minimum child weight (`min_child_weight`)**: Tested [1, 2, 3, 4, 5, 6]
  - Selected: **6** (regularization to prevent overfitting)

**Final XGBoost Parameters:**
```python
{
    'eta': 0.05,
    'max_depth': 3,
    'min_child_weight': 6,
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'aucpr'],
    'seed': 1
}
```

### Final Model Performance (Test Set)
- **ROC-AUC Score**: 0.693
- **Average Precision Score**: 0.144

---

## Files Description

| File | Description |
|------|-------------|
| `notebook.ipynb` | Jupyter notebook with EDA, model training, and evaluation |
| `predict.py` | Flask prediction service (exported training logic) |
| `train.py` | Training script (logic exported from notebook) |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Project configuration (uv package manager) |
| `Dockerfile` | Container configuration |
| `test-predict.py` | Local testing script |
| `test-predict-render.py` | Cloud testing script |
| `xgb_model_eta=0.05_depth=3_min-child=6_v0.0.bin` | Trained model + DictVectorizer |

---

## Reproducibility

### Getting the Data

**Option 1: Kaggle (Recommended)**
```bash
# Install kaggle CLI if needed
pip install kaggle

# Download the dataset
kaggle datasets download uciml/caravan-insurance-challenge
unzip caravan-insurance-challenge.zip -d data/
```

**Option 2: UCI Repository**
```bash
curl -o data/ticdata2000.txt https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/yourusername/caravan-insurance-capstone.git
cd caravan-insurance-capstone
```

2. Set up the environment (see Environment Setup below)

3. Launch Jupyter and run the notebook:
```bash
jupyter notebook notebook.ipynb
```

---

## Dependency and Environment Management

### Dependencies

All dependencies are specified in `requirements.txt`:
```
flask>=3.0.0
gunicorn>=21.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
numpy>=1.24.0
```

For development (notebook), additional dependencies are in `pyproject.toml`:
- pandas, matplotlib, seaborn, graphviz

### Environment Setup

**Option 1: Using pip with virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For notebook development, also install:
pip install pandas matplotlib seaborn jupyter graphviz
```

**Option 2: Using uv (fast package manager)**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

---

## Model Deployment

The model is deployed as a Flask REST API service.

### Running Locally

```bash
# Activate your virtual environment first
source venv/bin/activate

# Run the Flask service
python predict.py
```

The service will be available at `http://localhost:9696`

### API Endpoint

**POST** `/predict`

**Request Body** (JSON):
```json
{
    "MOSTYPE": 33,
    "MAANTHUI": 1,
    "MGEMOMV": 3,
    "MGEMLEEF": 2,
    "MOSHOOFD": 8,
    "MGODRK": 0,
    "MGODPR": 5,
    ...
}
```

**Response** (JSON):
```json
{
    "probability": 0.0083,
    "potential_customer": false
}
```

### Testing Locally

```bash
python test-predict.py
```

---

## Containerization

### Building the Docker Image

```bash
docker build -t caravan-insurance-prediction .
```

### Running the Container

```bash
# Run on default port 9696
docker run -p 9696:9696 caravan-insurance-prediction

# Or specify a custom port
docker run -e PORT=8080 -p 8080:8080 caravan-insurance-prediction
```

### Testing the Container

```bash
# In a separate terminal
python test-predict.py
```

### Dockerfile Details

The Dockerfile uses:
- Python 3.13 slim base image
- uv package manager for fast dependency installation
- Gunicorn WSGI server for production deployment
- Configurable PORT environment variable

---

## Cloud Deployment

The service is deployed on **Render** cloud platform.

### Live Service URL

🌐 **https://caravan-insurance-capstone.onrender.com**

### Testing the Live Service

```bash
python test-predict-render.py
```

**Note**: The first request may take 30-50 seconds if the instance is cold (free tier spins down after inactivity).

### Sample Output

```
Sending request to https://caravan-insurance-capstone.onrender.com/predict...
(First request may take 30-50 seconds if instance is cold)
{'potential_customer': False, 'probability': 0.008295536041259766}
Customer is unlikely to buy caravan insurance.
Probability: 0.83%
```

### Deploying to Render (Instructions)

1. Create a new **Web Service** on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure the service:
   - **Environment**: Docker
   - **Instance Type**: Free (or paid for always-on)
4. Render will automatically:
   - Build the Docker image
   - Deploy the service
   - Provide a public URL

---

## Project Structure

```
caravan-insurance-capstone/
├── data/
│   └── customer.json          # Sample customer for testing
├── notebook.ipynb             # EDA and model development
├── predict.py                 # Flask prediction service
├── requirements.txt           # Production dependencies
├── pyproject.toml            # Development dependencies (uv)
├── uv.lock                   # Locked dependencies
├── Dockerfile                # Container configuration
├── test-predict.py           # Local testing script
├── test-predict-render.py    # Cloud testing script
├── xgb_model_*.bin           # Trained model artifact
└── README.md                 # This file
```

---

## Future Improvements

- Implement threshold optimization for business-specific precision/recall trade-offs
- Add model monitoring and logging
- Implement A/B testing framework for model updates
- Add feature importance visualization endpoint
- Consider SMOTE or other resampling techniques for class imbalance
