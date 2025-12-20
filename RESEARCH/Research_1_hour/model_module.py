import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler


def convert_three_classes(y):
    s = y

    mu = s.mean()
    sigma = s.std()

    # Classes:
    # 0: neutral (within ±1σ of mean)
    # 1: positive (> +1σ)
    # 2: negative (< -1σ)

    y = np.where(
            s > mu + sigma, 1,
            np.where(s < mu - sigma, 2, 0)
        )
    return y


def model_preparation(X, y):
    # Step 1: 60% train, 40% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
    
    # Step 2: Split 40% into 50/50 → each gets 20% of original
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    print(y_val)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print(f"Test score - {accuracy_score(y_test, model.predict(X_test))}")
    print(f"Validation score - {accuracy_score(y_val, model.predict(X_val))}")
    print(f"Train score - {accuracy_score(y_train, model.predict(X_train))}")
    print(f"Train confusion matrix - {confusion_matrix(y_test,model.predict(X_test))}")
    print(classification_report(y_test,model.predict(X_test)))

    depth_hyperparams = range(1, 50, 2)
    training_acc = []
    validation_acc = []
    
    for d in depth_hyperparams:
        test_model = DecisionTreeClassifier(random_state=42, max_depth=d)
        test_model.fit(X_train, y_train)
        training_acc.append(test_model.score(X_train, y_train))
        validation_acc.append(test_model.score(X_val, y_val))
    
    # Print outside the loop (optional)
    print("Training Accuracy Scores:", training_acc[:3])
    print("Validation Accuracy Scores:", validation_acc[:3])
    
    # Return statement must be at the function level (not inside the for loop)
         
    fig, ax = plt.subplots(figsize=(8,5))  # Use subplots instead
    
    ax.plot(depth_hyperparams, training_acc, marker='o', label="Training Accuracy")
    ax.plot(depth_hyperparams, validation_acc, marker='o', label="Validation Accuracy")
    ax.set_xticks(depth_hyperparams)
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Training vs Validation Accuracy for Different Tree Depths")
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    # Remove plt.show() from here
    
    plt.show()



def best_model(X,y, max_depth=3):

    # Step 1: 60% train, 40% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)

    # Step 2: Split 40% into 50/50 → each gets 20% of original
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train,y_train)

    print(f"Test score - {accuracy_score(y_test,model.predict(X_test))}")
    print(f"Validation score - {accuracy_score(y_val,model.predict(X_val))}")
    print(f"Train score - {accuracy_score(y_train,model.predict(X_train))}")
    print(f"Train confusion matrix - {confusion_matrix(y_test,model.predict(X_test))}")
    print(classification_report(y_test,model.predict(X_test)))

    return model


from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    confusion_matrix, 
    classification_report,
    f1_score
)

def undersampler(X, y):
    """
    Undersample majority class (0 - Neutral) to balance with minority classes.
    Keeps all minority class samples.
    """
    rus = RandomUnderSampler(
        sampling_strategy='auto',  # Balance all classes to majority minority class
        random_state=42
    )
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(f"Original class distribution: {dict(pd.Series(y).value_counts().sort_index())}")
    print(f"Resampled class distribution: {dict(pd.Series(y_resampled).value_counts().sort_index())}")
    return X_resampled, y_resampled


def model_preparation_multi_class(X, y):
    """
    Train and tune DecisionTree classifier for multi-class imbalanced data.
    Uses balanced accuracy and proper train/val/test workflow.
    """
    # 60 / 20 / 20 split
    n = len(X)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print("\n=== DATA SPLITS ===")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Undersample only training data
    X_train, y_train = undersampler(X_train, y_train)

    # Baseline model (no max_depth constraint)
    print("\n=== BASELINE MODEL (No Depth Limit) ===")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print(f"Train Accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Train Balanced Accuracy: {balanced_accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Val Accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    print(f"Val Balanced Accuracy: {balanced_accuracy_score(y_val, model.predict(X_val)):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(f"Test Balanced Accuracy: {balanced_accuracy_score(y_test, model.predict(X_test)):.4f}")

    # Hyperparameter tuning: max_depth
    print("\n=== HYPERPARAMETER TUNING ===")
    depth_hyperparams = range(1, 50, 2)
    training_acc = []
    training_balanced_acc = []
    validation_acc = []
    validation_balanced_acc = []

    for d in depth_hyperparams:
        test_model = DecisionTreeClassifier(random_state=42, max_depth=d)
        test_model.fit(X_train, y_train)
        
        train_pred = test_model.predict(X_train)
        val_pred = test_model.predict(X_val)
        
        training_acc.append(accuracy_score(y_train, train_pred))
        training_balanced_acc.append(balanced_accuracy_score(y_train, train_pred))
        validation_acc.append(accuracy_score(y_val, val_pred))
        validation_balanced_acc.append(balanced_accuracy_score(y_val, val_pred))

    # Select best depth based on validation balanced accuracy
    best_idx = np.argmax(validation_balanced_acc)
    best_depth = depth_hyperparams[best_idx]
    best_val_balanced_acc = validation_balanced_acc[best_idx]

    print(f"Best max_depth: {best_depth}")
    print(f"Best validation balanced accuracy: {best_val_balanced_acc:.4f}")

    # Train final model with best depth
    print("\n=== FINAL MODEL (Optimized Depth) ===")
    final_model = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
    final_model.fit(X_train, y_train)

    # Final evaluation on test set
    test_pred = final_model.predict(X_test)
    
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test Balanced Accuracy: {balanced_accuracy_score(y_test, test_pred):.4f}")
    print(f"Test Macro F1: {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(f"Test Weighted F1: {f1_score(y_test, test_pred, average='weighted'):.4f}")
    
    print("\n=== CONFUSION MATRIX (Test Set) ===")
    print(confusion_matrix(y_test, test_pred))
    
    print("\n=== CLASSIFICATION REPORT (Test Set) ===")
    class_names = ['Neutral', 'Strong Up', 'Strong Down', 'Mid Up', 'Mid Down']
    print(classification_report(y_test, test_pred, target_names=class_names, digits=4))

    # Plot training vs validation curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Regular Accuracy
    ax1.plot(depth_hyperparams, training_acc, marker='o', label="Training Accuracy", linewidth=2)
    ax1.plot(depth_hyperparams, validation_acc, marker='o', label="Validation Accuracy", linewidth=2)
    ax1.axvline(x=best_depth, color='red', linestyle='--', label=f'Best Depth = {best_depth}')
    ax1.set_xlabel("Max Depth", fontsize=12)
    ax1.set_ylabel("Accuracy Score", fontsize=12)
    ax1.set_title("Training vs Validation Accuracy", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(range(1, 50, 5))

    # Plot 2: Balanced Accuracy
    ax2.plot(depth_hyperparams, training_balanced_acc, marker='o', label="Training Balanced Accuracy", linewidth=2)
    ax2.plot(depth_hyperparams, validation_balanced_acc, marker='o', label="Validation Balanced Accuracy", linewidth=2)
    ax2.axvline(x=best_depth, color='red', linestyle='--', label=f'Best Depth = {best_depth}')
    ax2.set_xlabel("Max Depth", fontsize=12)
    ax2.set_ylabel("Balanced Accuracy Score", fontsize=12)
    ax2.set_title("Training vs Validation Balanced Accuracy", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(range(1, 50, 5))

    plt.tight_layout()
    plt.show()

    return final_model, best_depth


# Example usage:
# X = df.drop(['target_multiclass', 'shifted_log_return'], axis=1)
# y = df['target_multiclass']
# final_model, best_depth = model_preparation_multi_class(X, y)
