import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load CSV file
data = pd.read_csv("fitness_data.csv")   

# Step 2: Features & Labels
X = data[['Age', 'Gender', 'Height_m', 'Weight_kg', 'Activity_Level']]
y = data['Label']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: BMI Calculator function
def calculate_bmi(height, weight):
    return weight / (height ** 2)

# Step 6: Prediction function
def predict_user():
    age = int(input("Enter Age: "))
    gender = int(input("Enter Gender (0=Female, 1=Male): "))
    height = float(input("Enter Height (m): "))
    weight = float(input("Enter Weight (kg): "))
    activity = int(input("Enter Activity Level (1-3): "))

    bmi = calculate_bmi(height, weight)

    # Prediction
    user_data = [[age, gender, height, weight, activity]]
    prediction = model.predict(user_data)[0]

    # Output
    print(f"\nYour BMI is: {bmi:.2f}")

    if prediction == 0:
        print("Category: Underfit")
        print("Recommendation: Increase calories, protein-rich diet, strength training")
    elif prediction == 1:
        print("Category: Fit")
        print("Recommendation: Maintain balanced diet and regular exercise")
    else:
        print("Category: Overfit")
        print("Recommendation: Calorie deficit, cardio + HIIT")

# Step 7: Run
predict_user()