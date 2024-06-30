import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'dataset.csv'
headers = [
    'base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z',
    'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z',
    'projected_gravity_x', 'projected_gravity_y', 'projected_gravity_z',
    'velocity_command_x', 'velocity_command_y', 'velocity_command_z',
    'joint_pos_1', 'joint_pos_2', 'joint_pos_3', 'joint_pos_4', 'joint_pos_5', 'joint_pos_6',
    'joint_pos_7', 'joint_pos_8', 'joint_pos_9', 'joint_pos_10', 'joint_pos_11', 'joint_pos_12',
    'joint_vel_1', 'joint_vel_2', 'joint_vel_3', 'joint_vel_4', 'joint_vel_5', 'joint_vel_6',
    'joint_vel_7', 'joint_vel_8', 'joint_vel_9', 'joint_vel_10', 'joint_vel_11', 'joint_vel_12',
    'joint_action_1', 'joint_action_2', 'joint_action_3', 'joint_action_4', 'joint_action_5', 'joint_action_6',
    'joint_action_7', 'joint_action_8', 'joint_action_9', 'joint_action_10', 'joint_action_11', 'joint_action_12'
]

# Read CSV
df = pd.read_csv(file_path, names=headers)

# Split the data into features and targets
X = df.iloc[:, 3:].values  # Using the remaining 45 columns as features
y = df.iloc[:, 0:3].values  # Predicting the first three columns

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the MLPRegressor model
mlp = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=500, random_state=42)

mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print model details
print(f"Model Coefficients: {mlp.coefs_}")
print(f"Model Intercepts: {mlp.intercepts_}")
