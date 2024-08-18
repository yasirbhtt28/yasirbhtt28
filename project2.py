import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Flood Prediction Model')
st.write("This application allows you to train models for flood prediction and visualize the results.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("First few rows of the dataset:")
    st.write(data.head())
    
    # Preprocessing
    data['Flood?'].fillna(0, inplace=True)
    columns_to_drop = ['Sl', 'Station_Names', 'Station_Number', 'X_COR', 'Y_COR', 'ALT', 'Period', 'Bright_Sunshine']
    data = data.drop(columns=columns_to_drop)
    
    # Splitting the dataset
    X = data.drop('Flood?', axis=1)
    y = data['Flood?']

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

    # Combine X_train and y_train for resampling
    train_data = pd.concat([X_train, y_train], axis=1)

    # Separate majority and minority classes
    majority_class = train_data[train_data['Flood?'] == 0]
    minority_class = train_data[train_data['Flood?'] == 1]

    # Downsample majority class
    majority_downsampled = resample(majority_class,
                                    replace=False,    # sample without replacement
                                    n_samples=len(minority_class), # to match minority class size
                                    random_state=42) # for reproducibility

    # Combine minority class with downsampled majority class
    downsampled = pd.concat([minority_class, majority_downsampled])

    # Separate X and y for the balanced dataset
    X_train_res = downsampled.drop('Flood?', axis=1)
    y_train_res = downsampled['Flood?']

    # Normalization and Scaling
    normalizer = MinMaxScaler()
    X_train_normalized = normalizer.fit_transform(X_train_res)
    X_test_normalized = normalizer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normalized)
    X_test_scaled = scaler.transform(X_test_normalized)

    # Model Training
    rf_model = RF(random_state=42)
    rf_model.fit(X_train_scaled, y_train_res)
    rf_predictions = rf_model.predict(X_test_scaled)

    lr_model = LR(random_state=42)
    lr_model.fit(X_train_scaled, y_train_res)
    lr_predictions = lr_model.predict(X_test_scaled)

    gb_model = GBC(random_state=42)
    gb_model.fit(X_train_scaled, y_train_res)
    gb_predictions = gb_model.predict(X_test_scaled)

    # Evaluation
    model_accuracies = {}

    def print_model_evaluation(model_name, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred)
        
        # Store accuracy in the dictionary
        model_accuracies[model_name] = accuracy
        
        # Print evaluation metrics
        st.write(f"### Model: {model_name}")
        st.write("**Accuracy:**", accuracy)
        st.write("**Precision:**", precision)
        st.write("**Recall:**", recall)
        st.write("**F1-Score:**", f1)
        st.write("**ROC AUC Score:**", auc)
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))
        st.write("-" * 50)

    # Print evaluations for each model
    print_model_evaluation("Random Forest", y_test, rf_predictions)
    print_model_evaluation("Logistic Regression", y_test, lr_predictions)
    print_model_evaluation("Gradient Boosting", y_test, gb_predictions)

    # Model accuracy comparison
    st.write("## Model Accuracy Comparison")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    st.pyplot(plt)

    # Function to plot confusion matrix
    def plot_confusion_matrix(y_test, y_pred, model_name):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

    # Plot confusion matrix for each model
    plot_confusion_matrix(y_test, rf_predictions, "Random Forest")
    plot_confusion_matrix(y_test, lr_predictions, "Logistic Regression")
    plot_confusion_matrix(y_test, gb_predictions, "Gradient Boosting")

    # ROC Curves for all models
    st.write("## ROC Curves")
    models = {
        "Random Forest": rf_model,
        "Logistic Regression": lr_model,
        "Gradient Boosting": gb_model
    }

    plt.figure(figsize=(14, 8))

    for model_name, model in models.items():
        y_pred_prob = model.predict_proba(X_test_scaled)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    st.pyplot(plt)
