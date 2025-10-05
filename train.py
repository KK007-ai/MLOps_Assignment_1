from sklearn.tree import DecisionTreeRegressor
import misc

# Main execution block
if __name__ == "__main__":
    # 1. Load data
    dataframe = misc.load_data()

    # 2. Preprocess data
    X_train, X_test, y_train, y_test = misc.preprocess_data(dataframe)

    # 3. Initialize the model
    dt_model = DecisionTreeRegressor(random_state=42)

    # 4. Train the model
    print("Training Decision Tree Regressor...")
    trained_dt_model = misc.train_model(dt_model, X_train, y_train)

    # 5. Evaluate the model
    misc.evaluate_model(trained_dt_model, X_test, y_test)