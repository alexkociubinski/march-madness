# run_predictor.py
from march import MarchMadnessPredictor

# Initialize the predictor
predictor = MarchMadnessPredictor()

# Load team data (in a real scenario, you'd have real team names)
predictor.scrape_team_data(year=2025)

# Load historical data and train the model
predictor.load_historical_data()
predictor.train_model()

# Create a sample bracket with team names
# In a real scenario, you'd use actual team names
bracket = {
    'East': [f"Team_{i}" for i in range(1, 17)],
    'West': [f"Team_{i+16}" for i in range(1, 17)],
    'Midwest': [f"Team_{i+32}" for i in range(1, 17)],
    'South': [f"Team_{i+48}" for i in range(1, 17)]
}

# Simulate the tournament 100 times
results = predictor.simulate_tournament(bracket, num_simulations=100)

# Print the bracket
predictor.print_bracket(results)

# Create visualizations
predictor.visualize_bracket(results)