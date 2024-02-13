def prb1() :

    def maxIN( arr ) :
        print( f"Maximum in array : {arr} is {max(arr)}")

    def minIN( arr ) :
        print( f"Minimum in array : {arr} is {min(arr)}")

    def sumOfElements( arr ) :
        print( f"Sum of elements : {arr} is {sum(arr)}" )

    def avg( arr ) :
        print( f"Average of array : {arr} is {sum(arr)/len(arr)}")

    def squareRoot( arr ) :
        print( f"Square root of {arr[-1]} is {arr[-1]**0.5}" )
        print( f"Round-Off value {round(arr[-1]**0.5)}" )

    arr = list(map(int, input("Enter array:").split()))
    if len(arr)>0 :
        maxIN(arr)
        minIN(arr)
        sumOfElements(arr)
        avg(arr)
        squareRoot(arr)
    else :
        print("Empty array")

def prb2() :
    import numpy as np

    def findMean( arr ) :
        print( f"Mean is : {np.mean(arr)}")

    def findMedian( arr ) :
        print( f"Median is : {np.median(arr)}")

    def findMode( arr ) :
        unique_elements, counts = np.unique(arr, return_counts=True)
        mode_index = np.argmax(counts)
        mode = unique_elements[mode_index]
        print( f"Mode is : {mode}")

    def findStandardDeviation( arr ) :
        print( f"Standard Deviation is : {np.std(arr)}")

    arr = list(map(int, input("Enter array:").split()))
    if len(arr)>0 :
        print( f"For array {arr}")
        findMean(arr)
        findMedian(arr)
        findMode(arr)
        findStandardDeviation(arr)
    else :
        print("Empty array")

def prb3() :
    import pandas as pd

    data = {
        "Name" : [ 'Willam', 'Willam', 'Emma', 'Emma', 'Anika', 'Anika'],
        "Region" : [ 'East', 'East', 'North', 'West', 'East', 'East'],
        "Sales" : [ 50000, 50000, 52000, 52000, 65000, 72000],
        "Expense" : [42000, 42000, 43000, 43000, 44000, 53000]
    }
    df = pd.DataFrame( data )
    print( df[ ["Region", "Sales" ]] )
    print(df.loc[1,"Sales"])
    print(df.iloc[1,2])
    new_row = { "Name" : "ABC", "Region" : "South", "Sales" : 80000, "Expense" : 40000 }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(df)
    df['Gender'] = ['Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Male']
    print(df)
    df.rename(columns={'Name': 'Ename'}, inplace=True)
    print(df)
    print(df.groupby(["Gender"]).sum())
    print(df.groupby(["Gender"]).mean())
    print(df.groupby(["Gender"]).std())

def prb4a() :
    import pandas as pd
    import numpy as np

    data = {'Name': ['Alice', 'Bob', np.nan, 'Charlie', 'David'],
            'Age': [25, 30, np.nan, 35, 40],
            'City': ['New York', 'San Francisco', 'Los Angeles', np.nan, 'Seattle']}

    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nMissing Values:")
    print(df.isnull())
    df_dropped = df.dropna()
    df_filled = df.fillna(value='Unknown')
    numeric_columns = df.select_dtypes(include=np.number).columns
    df_mean_filled = df.copy()
    df_mean_filled[numeric_columns] = df_mean_filled[numeric_columns].fillna(df_mean_filled[numeric_columns].mean())
    print("\nDataFrame after dropping rows with missing values:")
    print(df_dropped)
    print("\nDataFrame after filling missing values with 'Unknown':")
    print(df_filled)
    print("\nDataFrame after filling missing values with the mean of numeric columns:")
    print(df_mean_filled)

def prb4b() :

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    data = {'Feature1': [10, 20, 30, 40, 50],
            'Feature2': [5, 15, 25, 35, 45]}
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    scaler = MinMaxScaler()

    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

    print("\nDataFrame after Min-Max normalization:")
    print(normalized_df)

def prb5() :
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    # Load the diabetes dataset from scikit-learn
    from sklearn.datasets import load_diabetes

    # Load the diabetes dataset
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # SVM classifier without dimensionality reduction
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train_std, y_train)
    y_pred = svm_classifier.predict(X_test_std)
    accuracy_no_reduction = accuracy_score(y_test, y_pred)
    print(f"Accuracy without dimensionality reduction: {accuracy_no_reduction}")

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # SVM classifier with dimensionality reduction
    svm_classifier_pca = SVC(kernel='linear', random_state=42)
    svm_classifier_pca.fit(X_train_pca, y_train)
    y_pred_pca = svm_classifier_pca.predict(X_test_pca)
    accuracy_with_reduction = accuracy_score(y_test, y_pred_pca)
    print(f"Accuracy with dimensionality reduction: {accuracy_with_reduction}")

def prb6() :
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Generating a synthetic dataset
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Visualize the results
    plt.scatter(X_test, y_test, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression Model')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.title('Simple Linear Regression Example')
    plt.legend()
    plt.show()

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')

def prb7() :
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    # Generate sample data for demonstration
    data, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # Function to calculate the sum of squared distances (inertia) for different values of k
    def calculate_inertia(data, k_range):
        inertia_values = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertia_values.append(kmeans.inertia_)
        return inertia_values

    # Specify the range of k values to test
    k_values = range(1, 11)

    # Calculate inertia for each value of k
    inertia_values = calculate_inertia(data, k_values)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.grid(True)
    plt.show()

def prb8() :
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    import pandas as pd

    # Sample transaction data (replace this with your dataset)
    data = {'Transaction': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
            'Item': ['A', 'B', 'C', 'A', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'D']}
    df = pd.DataFrame(data)

    # Convert data to one-hot encoded format
    basket = pd.crosstab(df['Transaction'], df['Item'], dropna=False)
    basket = (basket > 0).astype(int)

    # Apply Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    # Display the results
    print("Frequent Itemsets:")
    print(frequent_itemsets)

    print("\nAssociation Rules:")
    print(rules)

def prb9() :
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix
    from mlxtend.plotting import plot_decision_regions  # Install mlxtend with: pip install mlxtend

    # Load the Iris dataset for demonstration purposes
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # Use only the first two features for visualization
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Function to train and evaluate a classifier, and visualize the decision boundary
    def train_evaluate_visualize_classifier(classifier, X_train, y_train, X_test, y_test, title):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        plt.figure(figsize=(8, 6))
        plot_decision_regions(X_train, y_train, clf=classifier, legend=2)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.show()

    # Logistic Regression
    logreg = LogisticRegression(random_state=42)
    train_evaluate_visualize_classifier(logreg, X_train_std, y_train, X_test_std, y_test, "Logistic Regression")

    # k-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=5)
    train_evaluate_visualize_classifier(knn, X_train_std, y_train, X_test_std, y_test, "k-Nearest Neighbors (KNN)")

    # Naive Bayes
    nb = GaussianNB()
    train_evaluate_visualize_classifier(nb, X_train_std, y_train, X_test_std, y_test, "Naive Bayes")

    # Support Vector Machine (SVM)
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    train_evaluate_visualize_classifier(svm, X_train_std, y_train, X_test_std, y_test, "Support Vector Machine (SVM)")

