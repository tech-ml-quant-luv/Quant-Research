import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def model_preparation(X, y):
    # Step 1: 60% train, 40% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)

    # Step 2: Split 40% into 50/50 → each gets 20% of original
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print(f"Test score - {accuracy_score(y_test, model.predict(X_test))}")
    print(f"Validation score - {accuracy_score(y_val, model.predict(X_val))}")
    print(f"Train score - {accuracy_score(y_train, model.predict(X_train))}")
    print(f"Train confusion matrix - {confusion_matrix(y_train, model.predict(X_train))}")
    print(classification_report(y_train, model.predict(X_train)))

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
    print(f"Validation score - {accuracy_score(y_train,model.predict(X_train))}")
    print(f"Train confusion matrix - {confusion_matrix(y_train,model.predict(X_train))}")
    print(classification_report(y_train,model.predict(X_train)
    ))

    return model


