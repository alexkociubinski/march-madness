import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import random
import matplotlib.pyplot as plt
import seaborn as sns

class MarchMadnessPredictor:
    def __init__(self):
        self.teams_data = {}
        # Use Gradient Boosting for better handling of non-linear relationships
        self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        self.upset_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        self.scaler = StandardScaler()
        self.training_data = None
        self.training_labels = None
        self.historical_upsets = None
        
    def scrape_team_data(self, year=2025):
        """
        Scrape team statistics from sports websites
        """
        print(f"Collecting team data for {year}...")
        
        # In a real implementation, you'd use APIs or web scraping
        # Advanced stats to collect:
        advanced_features = [
            # Offensive metrics
            'adjusted_offensive_efficiency', 
            'effective_field_goal_percentage',
            'turnover_percentage',
            'offensive_rebound_percentage',
            'free_throw_rate',
            'three_point_percentage',
            'two_point_percentage',
            'free_throw_percentage',
            'points_per_possession',
            
            # Defensive metrics
            'adjusted_defensive_efficiency',
            'opponent_effective_field_goal_percentage',
            'opponent_turnover_percentage',
            'defensive_rebound_percentage',
            'opponent_free_throw_rate',
            'opponent_three_point_percentage',
            'opponent_two_point_percentage',
            'blocks_per_game',
            'steals_per_game',
            
            # Team metrics
            'tempo',
            'average_height',
            'experience_years',
            'bench_minutes_percentage',
            'strength_of_schedule',
            'non_conference_sos',
            'conference_sos',
            'wins_against_top_25',
            'losses_against_top_25',
            'conference_record',
            'road_record',
            'neutral_court_record',
            'last_10_games_record',
            
            # Tournament history
            'tourney_appearances_last_5_years',
            'final_four_appearances_last_10_years',
            'upset_wins_last_5_years',
            'upset_losses_last_5_years',
            
            # Momentum metrics
            'win_streak',
            'conference_tournament_result',
            'days_since_last_game',
            
            # Advanced analytics
            'kenpom_ranking',
            'barttorvik_ranking',
            'sagarin_rating',
            'net_ranking',
            'bpi_ranking'
        ]
        
        # For demo purposes, we'll create randomized data
        all_teams = [f"Team_{i}" for i in range(1, 65)]
        for team in all_teams:
            self.teams_data[team] = {
                feature: random.uniform(0, 100) for feature in advanced_features
            }
            # Add seed as a feature (1-16)
            self.teams_data[team]['seed'] = random.randint(1, 16)
            
            # Add conference
            conferences = ['ACC', 'Big Ten', 'Big 12', 'SEC', 'Pac-12', 'Big East', 'American', 'Atlantic 10', 'Mountain West', 'WCC']
            self.teams_data[team]['conference'] = random.choice(conferences)
            
            # Add coach experience
            self.teams_data[team]['coach_experience_years'] = random.randint(1, 30)
            self.teams_data[team]['coach_tournament_wins'] = random.randint(0, 50)
            
        return self.teams_data
    
    def load_historical_data(self, years_range=(2010, 2024)):
        """
        Load historical tournament results to train the model
        """
        print(f"Loading historical data from {years_range[0]} to {years_range[1]}...")
        
        # In a real implementation, you'd load CSV files or use an API
        # Generate synthetic data with realistic patterns
        
        features = []
        results = []
        upset_indicators = []
        
        # Generate synthetic historical data
        for _ in range(2000):  # 2000 synthetic games for more robust training
            # Team A is typically the higher seed
            team_a_seed = random.randint(1, 16)
            team_b_seed = random.randint(team_a_seed, 16)  # B is always equal or lower seeded
            
            # Create feature vector for matchup
            # Include a comprehensive set of differentials
            feature_vector = [
                # Seed and ranking differentials
                team_a_seed - team_b_seed,  # Negative means team A is higher seed
                random.uniform(-30, 30),  # kenpom ranking differential
                random.uniform(-30, 30),  # barttorvik ranking differential
                
                # Offensive differentials
                random.uniform(-20, 20),  # Offensive efficiency
                random.uniform(-15, 15),  # Effective FG%
                random.uniform(-15, 15),  # Turnover %
                random.uniform(-15, 15),  # Off. rebound %
                random.uniform(-15, 15),  # Free throw rate
                random.uniform(-15, 15),  # 3-point %
                
                # Defensive differentials
                random.uniform(-20, 20),  # Defensive efficiency
                random.uniform(-15, 15),  # Opponent eFG%
                random.uniform(-15, 15),  # Opponent turnover %
                random.uniform(-15, 15),  # Def. rebound %
                
                # Team metrics differentials
                random.uniform(-10, 10),  # Tempo
                random.uniform(-10, 10),  # Experience
                random.uniform(-10, 10),  # Strength of schedule
                random.uniform(-15, 15),  # Wins against top 25
                
                # Tournament experience
                random.uniform(-5, 5),  # Tournament appearances
                random.uniform(-5, 5),  # Final four appearances
                
                # Momentum 
                random.uniform(-10, 10),  # Win streak
                random.uniform(-3, 3),    # Last 10 games record
                
                # Coach experience
                random.uniform(-20, 20),  # Coach experience years
                random.uniform(-30, 30),  # Coach tournament wins
            ]
            
            # Determine outcome based on seed difference and other factors
            # Base probability from seed difference
            seed_diff = team_b_seed - team_a_seed
            base_prob = 0.5 + (seed_diff * 0.05)  # Higher seed has advantage
            
            # Adjust for other factors
            quality_diff = sum(feature_vector[1:]) / 100  # Overall quality difference
            momentum_factor = (feature_vector[-1] + feature_vector[-2]) / 50  # Momentum and coaching
            
            # Final probability
            win_probability = base_prob + quality_diff + momentum_factor
            win_probability = max(0.1, min(0.9, win_probability))  # Cap between 10% and 90%
            
            # Determine result
            result = 1 if random.random() < win_probability else 0
            
            # Determine if this is an upset
            is_upset = 1 if result == 0 and seed_diff > 0 else 0
            
            features.append(feature_vector)
            results.append(result)
            upset_indicators.append(is_upset)
        
        self.training_data = np.array(features)
        self.training_labels = np.array(results)
        self.historical_upsets = np.array(upset_indicators)
        
        return self.training_data, self.training_labels, self.historical_upsets
    
    def analyze_upsets(self):
        """
        Analyze historical upsets to identify patterns
        """
        if self.training_data is None or self.historical_upsets is None:
            self.load_historical_data()
            
        print("Analyzing historical upset patterns...")
        
        # Train a dedicated model for predicting upsets
        upset_indices = np.where(self.historical_upsets == 1)[0]
        non_upset_indices = np.where(self.historical_upsets == 0)[0]
        
        # Balance the dataset for better upset prediction
        n_upsets = len(upset_indices)
        non_upset_sample = np.random.choice(non_upset_indices, size=n_upsets*2, replace=False)
        
        # Combine upset and non-upset samples
        balanced_indices = np.concatenate([upset_indices, non_upset_sample])
        
        X_balanced = self.training_data[balanced_indices]
        y_balanced = self.historical_upsets[balanced_indices]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_balanced)
        
        # Train the upset prediction model
        self.upset_model.fit(X_scaled, y_balanced)
        
        # Analyze feature importance for upsets
        feature_importance = self.upset_model.feature_importances_
        
        # Return top factors that contribute to upsets
        return feature_importance
    
    def train_model(self):
        """
        Train the prediction model using historical data
        """
        if self.training_data is None:
            self.load_historical_data()
            
        print("Training prediction model...")
        
        # Split data into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_data, self.training_labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Model trained with accuracy: {train_score:.4f} on training data, {test_score:.4f} on test data")
        
        # Train the upset model
        self.analyze_upsets()
        
        return train_score, test_score
    
    def feature_vector_from_teams(self, team_a, team_b):
        """
        Create a feature vector for a matchup between two teams
        """
        if not self.teams_data:
            self.scrape_team_data()
            
        # Extract features for both teams
        team_a_data = self.teams_data[team_a]
        team_b_data = self.teams_data[team_b]
        
        # Create feature vector (differences between teams)
        # Match the same features used in training
        features = [
            # Seed and ranking differentials
            team_a_data['seed'] - team_b_data['seed'],
            team_a_data['kenpom_ranking'] - team_b_data['kenpom_ranking'],
            team_a_data['barttorvik_ranking'] - team_b_data['barttorvik_ranking'],
            
            # Offensive differentials
            team_a_data['adjusted_offensive_efficiency'] - team_b_data['adjusted_offensive_efficiency'],
            team_a_data['effective_field_goal_percentage'] - team_b_data['effective_field_goal_percentage'],
            team_a_data['turnover_percentage'] - team_b_data['turnover_percentage'],
            team_a_data['offensive_rebound_percentage'] - team_b_data['offensive_rebound_percentage'],
            team_a_data['free_throw_rate'] - team_b_data['free_throw_rate'],
            team_a_data['three_point_percentage'] - team_b_data['three_point_percentage'],
            
            # Defensive differentials
            team_a_data['adjusted_defensive_efficiency'] - team_b_data['adjusted_defensive_efficiency'],
            team_a_data['opponent_effective_field_goal_percentage'] - team_b_data['opponent_effective_field_goal_percentage'],
            team_a_data['opponent_turnover_percentage'] - team_b_data['opponent_turnover_percentage'],
            team_a_data['defensive_rebound_percentage'] - team_b_data['defensive_rebound_percentage'],
            
            # Team metrics differentials
            team_a_data['tempo'] - team_b_data['tempo'],
            team_a_data['experience_years'] - team_b_data['experience_years'],
            team_a_data['strength_of_schedule'] - team_b_data['strength_of_schedule'],
            team_a_data['wins_against_top_25'] - team_b_data['wins_against_top_25'],
            
            # Tournament experience
            team_a_data['tourney_appearances_last_5_years'] - team_b_data['tourney_appearances_last_5_years'],
            team_a_data['final_four_appearances_last_10_years'] - team_b_data['final_four_appearances_last_10_years'],
            
            # Momentum 
            team_a_data['win_streak'] - team_b_data['win_streak'],
            team_a_data['last_10_games_record'] - team_b_data['last_10_games_record'],
            
            # Coach experience
            team_a_data['coach_experience_years'] - team_b_data['coach_experience_years'],
            team_a_data['coach_tournament_wins'] - team_b_data['coach_tournament_wins'],
        ]
        
        return features
    
    def predict_matchup(self, team_a, team_b):
        """
        Predict the winner of a matchup between two teams
        """
        if not self.teams_data:
            self.scrape_team_data()
            
        # Create feature vector
        features = self.feature_vector_from_teams(team_a, team_b)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict winner
        prediction = self.model.predict_proba(features_scaled)[0]
        
        # Predict upset probability
        upset_probability = self.upset_model.predict_proba(features_scaled)[0][1]
        
        # Determine if this is an upset scenario
        is_potential_upset = self.teams_data[team_a]['seed'] > self.teams_data[team_b]['seed']
        
        # Adjust prediction based on upset model
        if is_potential_upset and upset_probability > 0.35:
            # Increase the chance of an upset
            adjustment = min(0.15, upset_probability * 0.3)
            adjusted_prediction = [prediction[0] + adjustment, prediction[1] - adjustment]
        else:
            adjusted_prediction = prediction
        
        return {
            'team_a_win_probability': adjusted_prediction[1],
            'team_b_win_probability': adjusted_prediction[0],
            'upset_probability': upset_probability if is_potential_upset else 1 - upset_probability,
            'predicted_winner': team_a if adjusted_prediction[1] > adjusted_prediction[0] else team_b
        }
    
    def simulate_tournament(self, bracket, num_simulations=100):
        """
        Simulate the entire tournament multiple times to account for variability
        bracket: dictionary mapping regions to lists of teams in seed order
        """
        print(f"Simulating tournament {num_simulations} times...")
        
        # Track championship counts
        champion_counts = {}
        final_four_counts = {}
        
        for sim in range(num_simulations):
            if sim % 10 == 0:
                print(f"Running simulation {sim + 1}/{num_simulations}")
                
            # First round matchups (1 vs 16, 2 vs 15, etc.)
            first_round = {}
            for region, teams in bracket.items():
                first_round[region] = []
                for i in range(8):
                    team_a = teams[i]
                    team_b = teams[15-i]
                    
                    # Add randomness to simulations
                    prediction = self.predict_matchup(team_a, team_b)
                    rand_val = random.random()
                    
                    # Determine winner based on probabilities
                    winner = team_a if rand_val < prediction['team_a_win_probability'] else team_b
                    first_round[region].append(winner)
            
            # Second round
            second_round = {}
            for region, winners in first_round.items():
                second_round[region] = []
                for i in range(0, 8, 2):
                    team_a = winners[i]
                    team_b = winners[i+1]
                    prediction = self.predict_matchup(team_a, team_b)
                    rand_val = random.random()
                    winner = team_a if rand_val < prediction['team_a_win_probability'] else team_b
                    second_round[region].append(winner)
                    
            # Sweet 16
            sweet_16 = {}
            for region, winners in second_round.items():
                sweet_16[region] = []
                for i in range(0, 4, 2):
                    team_a = winners[i]
                    team_b = winners[i+1]
                    prediction = self.predict_matchup(team_a, team_b)
                    rand_val = random.random()
                    winner = team_a if rand_val < prediction['team_a_win_probability'] else team_b
                    sweet_16[region].append(winner)
                    
            # Elite 8
            elite_8 = {}
            for region, winners in sweet_16.items():
                team_a = winners[0]
                team_b = winners[1]
                prediction = self.predict_matchup(team_a, team_b)
                rand_val = random.random()
                winner = team_a if rand_val < prediction['team_a_win_probability'] else team_b
                elite_8[region] = winner
                
                # Count Final Four appearances
                final_four_counts[winner] = final_four_counts.get(winner, 0) + 1
                    
            # Final 4
            regions = list(elite_8.keys())
            final_4_matchup_1 = self.predict_matchup(elite_8[regions[0]], elite_8[regions[1]])
            final_4_matchup_2 = self.predict_matchup(elite_8[regions[2]], elite_8[regions[3]])
            
            rand_val_1 = random.random()
            rand_val_2 = random.random()
            
            finalist_1 = elite_8[regions[0]] if rand_val_1 < final_4_matchup_1['team_a_win_probability'] else elite_8[regions[1]]
            finalist_2 = elite_8[regions[2]] if rand_val_2 < final_4_matchup_2['team_a_win_probability'] else elite_8[regions[3]]
            
            # Championship
            championship = self.predict_matchup(finalist_1, finalist_2)
            rand_val = random.random()
            champion = finalist_1 if rand_val < championship['team_a_win_probability'] else finalist_2
            
            # Count championship wins
            champion_counts[champion] = champion_counts.get(champion, 0) + 1
        
        # Get most common outcomes
        champion_probs = {team: count/num_simulations for team, count in champion_counts.items()}
        final_four_probs = {team: count/num_simulations for team, count in final_four_counts.items()}
        
        # Sort by probability
        champion_probs = dict(sorted(champion_probs.items(), key=lambda x: x[1], reverse=True))
        final_four_probs = dict(sorted(final_four_probs.items(), key=lambda x: x[1], reverse=True))
        
        # Simulate one full bracket for display
        results = self.simulate_single_tournament(bracket)
        
        return {
            'bracket': results,
            'champion_probabilities': champion_probs,
            'final_four_probabilities': final_four_probs
        }
    
    def simulate_single_tournament(self, bracket):
        """
        Simulate a single tournament run
        """
        results = {}
        
        # First round matchups (1 vs 16, 2 vs 15, etc.)
        first_round = {}
        for region, teams in bracket.items():
            first_round[region] = []
            for i in range(8):
                team_a = teams[i]
                team_b = teams[15-i]
                prediction = self.predict_matchup(team_a, team_b)
                winner = prediction['predicted_winner']
                upset = (team_b == winner and self.teams_data[team_a]['seed'] < self.teams_data[team_b]['seed'])
                first_round[region].append({
                    'winner': winner,
                    'upset': upset,
                    'probability': prediction['team_a_win_probability'] if winner == team_a else prediction['team_b_win_probability']
                })
        
        results['first_round'] = first_round
        
        # Second round
        second_round = {}
        for region, winners in first_round.items():
            second_round[region] = []
            for i in range(0, 8, 2):
                team_a = winners[i]['winner']
                team_b = winners[i+1]['winner']
                prediction = self.predict_matchup(team_a, team_b)
                winner = prediction['predicted_winner']
                upset = (team_b == winner and self.teams_data[team_a]['seed'] < self.teams_data[team_b]['seed'])
                second_round[region].append({
                    'winner': winner,
                    'upset': upset,
                    'probability': prediction['team_a_win_probability'] if winner == team_a else prediction['team_b_win_probability']
                })
                
        results['second_round'] = second_round
        
        # Sweet 16
        sweet_16 = {}
        for region, winners in second_round.items():
            sweet_16[region] = []
            for i in range(0, 4, 2):
                team_a = winners[i]['winner']
                team_b = winners[i+1]['winner']
                prediction = self.predict_matchup(team_a, team_b)
                winner = prediction['predicted_winner']
                upset = (team_b == winner and self.teams_data[team_a]['seed'] < self.teams_data[team_b]['seed'])
                sweet_16[region].append({
                    'winner': winner,
                    'upset': upset,
                    'probability': prediction['team_a_win_probability'] if winner == team_a else prediction['team_b_win_probability']
                })
                
        results['sweet_16'] = sweet_16
        
        # Elite 8
        elite_8 = {}
        for region, winners in sweet_16.items():
            team_a = winners[0]['winner']
            team_b = winners[1]['winner']
            prediction = self.predict_matchup(team_a, team_b)
            winner = prediction['predicted_winner']
            upset = (team_b == winner and self.teams_data[team_a]['seed'] < self.teams_data[team_b]['seed'])
            elite_8[region] = {
                'winner': winner,
                'upset': upset,
                'probability': prediction['team_a_win_probability'] if winner == team_a else prediction['team_b_win_probability']
            }
                
        results['elite_8'] = elite_8
        
        # Final 4
        regions = list(elite_8.keys())
        team_a = elite_8[regions[0]]['winner']
        team_b = elite_8[regions[1]]['winner']
        team_c = elite_8[regions[2]]['winner']
        team_d = elite_8[regions[3]]['winner']
        
        final_4_matchup_1 = self.predict_matchup(team_a, team_b)
        final_4_matchup_2 = self.predict_matchup(team_c, team_d)
        
        finalist_1 = final_4_matchup_1['predicted_winner']
        upset_1 = (team_b == finalist_1 and self.teams_data[team_a]['seed'] < self.teams_data[team_b]['seed'])
        
        finalist_2 = final_4_matchup_2['predicted_winner']
        upset_2 = (team_d == finalist_2 and self.teams_data[team_c]['seed'] < self.teams_data[team_d]['seed'])
        
        results['final_4'] = {
            'matchup_1': {
                'teams': [team_a, team_b],
                'winner': finalist_1,
                'upset': upset_1,
                'probability': final_4_matchup_1['team_a_win_probability'] if finalist_1 == team_a else final_4_matchup_1['team_b_win_probability']
            },
            'matchup_2': {
                'teams': [team_c, team_d],
                'winner': finalist_2,
                'upset': upset_2,
                'probability': final_4_matchup_2['team_a_win_probability'] if finalist_2 == team_c else final_4_matchup_2['team_b_win_probability']
            }
        }
        
        # Championship
        championship = self.predict_matchup(finalist_1, finalist_2)
        champion = championship['predicted_winner']
        upset_final = (finalist_2 == champion and self.teams_data[finalist_1]['seed'] < self.teams_data[finalist_2]['seed'])
        
        results['championship'] = {
            'finalists': [finalist_1, finalist_2],
            'champion': champion,
            'upset': upset_final,
            'win_probability': championship['team_a_win_probability'] if champion == finalist_1 else championship['team_b_win_probability']
        }
        
        return results
    
    def print_bracket(self, results):
        """
        Print the bracket in a readable format with upset indicators
        """
        print("\n===== MARCH MADNESS BRACKET PREDICTION =====\n")
        
        # Print first round
        print("FIRST ROUND WINNERS:")
        for region, winners in results['bracket']['first_round'].items():
            print(f"  {region}:")
            for i, match in enumerate(winners):
                upset_tag = "⚠️ UPSET" if match['upset'] else ""
                print(f"    {match['winner']} (Win Prob: {match['probability']:.2%}) {upset_tag}")
        
        # Print second round
        print("\nSECOND ROUND WINNERS:")
        for region, winners in results['bracket']['second_round'].items():
            print(f"  {region}:")
            for match in winners:
                upset_tag = "⚠️ UPSET" if match['upset'] else ""
                print(f"    {match['winner']} (Win Prob: {match['probability']:.2%}) {upset_tag}")
        
        # Print Sweet 16
        print("\nSWEET 16 WINNERS:")
        for region, winners in results['bracket']['sweet_16'].items():
            print(f"  {region}:")
            for match in winners:
                upset_tag = "⚠️ UPSET" if match['upset'] else ""
                print(f"    {match['winner']} (Win Prob: {match['probability']:.2%}) {upset_tag}")
        
        # Print Elite 8
        print("\nELITE 8 WINNERS (REGIONAL CHAMPIONS):")
        for region, match in results['bracket']['elite_8'].items():
            upset_tag = "⚠️ UPSET" if match['upset'] else ""
            print(f"  {region}: {match['winner']} (Win Prob: {match['probability']:.2%}) {upset_tag}")
        
        # Print Final Four
        print("\nFINAL FOUR:")
        final_4 = results['bracket']['final_4']
        matchup_1 = final_4['matchup_1']
        matchup_2 = final_4['matchup_2']
        
        upset_tag_1 = "⚠️ UPSET" if matchup_1['upset'] else ""
        upset_tag_2 = "⚠️ UPSET" if matchup_2['upset'] else ""
        
        print(f"  Matchup 1: {matchup_1['teams'][0]} vs {matchup_1['teams'][1]} → {matchup_1['winner']} (Win Prob: {matchup_1['probability']:.2%}) {upset_tag_1}")
        print(f"  Matchup 2: {matchup_2['teams'][0]} vs {matchup_2['teams'][1]} → {matchup_2['winner']} (Win Prob: {matchup_2['probability']:.2%}) {upset_tag_2}")
        
        # Print Championship
        print("\nCHAMPIONSHIP:")
        championship = results['bracket']['championship']
        upset_tag = "⚠️ UPSET" if championship['upset'] else ""
        print(f"  {championship['finalists'][0]} vs {championship['finalists'][1]} → {championship['champion']} (Win Prob: {championship['win_probability']:.2%}) {upset_tag}")
        
        print(f"\n🏆 PREDICTED CHAMPION: {championship['champion']} 🏆")
        
        # Print most likely champions
        print("\nMOST LIKELY CHAMPIONS (BASED ON MULTIPLE SIMULATIONS):")
        for i, (team, prob) in enumerate(list(results['champion_probabilities'].items())[:5]):
            print(f"  {i+1}. {team}: {prob:.2%}")
        
        # Print most likely Final Four teams
        print("\nMOST LIKELY FINAL FOUR TEAMS:")
        for i, (team, prob) in enumerate(list(results['final_four_probabilities'].items())[:8]):
            print(f"  {i+1}. {team}: {prob:.2%}")
    
def visualize_bracket(self, results=None):
    """
    Create visualizations of the bracket and predictions
    
    Parameters:
    results: Dictionary containing simulation results. If None, will run a new simulation.
    """
    # Check if results are provided, if not, run a simulation
    if results is None:
        # Create a sample bracket for simulation
        sample_bracket = {
            'East': [f"Team_{i}" for i in range(1, 17)],
            'West': [f"Team_{i+16}" for i in range(1, 17)],
            'Midwest': [f"Team_{i+32}" for i in range(1, 17)],
            'South': [f"Team_{i+48}" for i in range(1, 17)]
        }
        
        print("No results provided. Running tournament simulation...")
        results = self.simulate_tournament(sample_bracket, num_simulations=50)
    
    # Validate that results dictionary has the expected structure
    required_keys = ['bracket', 'champion_probabilities', 'final_four_probabilities']
    if not all(key in results for key in required_keys):
        raise ValueError("Results dictionary is missing required keys. Expected keys: " + 
                        ", ".join(required_keys))
    
    # Validate bracket structure
    bracket_rounds = ['first_round', 'second_round', 'sweet_16', 'elite_8', 'final_4', 'championship']
    if not all(round_name in results['bracket'] for round_name in bracket_rounds):
        raise ValueError("Bracket results are missing one or more tournament rounds")
    
    # Now create visualizations
    # Create a figure with champion probabilities
    plt.figure(figsize=(12, 8))
    
    # Plot top 10 champion probabilities
    top_champions = list(results['champion_probabilities'].items())[:10]
    teams = [team for team, _ in top_champions]
    probs = [prob for _, prob in top_champions]
    
    plt.barh(teams, probs, color='skyblue')
    plt.xlabel('Probability')
    plt.ylabel('Team')
    plt.title('Championship Probabilities')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, prob in enumerate(probs):
        plt.text(prob + 0.01, i, f'{prob:.1%}')
    
    plt.tight_layout()
    plt.savefig('championship_probabilities.png')
    plt.close()
    
    # Create a figure with Final Four probabilities
    plt.figure(figsize=(12, 10))
    
    # Plot top 16 Final Four probabilities
    top_final_four = list(results['final_four_probabilities'].items())[:16]
    ff_teams = [team for team, _ in top_final_four]
    ff_probs = [prob for _, prob in top_final_four]
    
    plt.barh(ff_teams, ff_probs, color='lightgreen')
    plt.xlabel('Probability')
    plt.ylabel('Team')
    plt.title('Final Four Probabilities')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, prob in enumerate(ff_probs):
        plt.text(prob + 0.01, i, f'{prob:.1%}')
    
    plt.tight_layout()
    plt.savefig('final_four_probabilities.png')
    plt.close()
    
    # Create a heatmap of upsets in the bracket
    plt.figure(figsize=(14, 8))
    
    # Extract upset data from each round
    rounds = ['first_round', 'second_round', 'sweet_16', 'elite_8']
    regions = list(results['bracket']['elite_8'].keys())
    
    # Create upset matrix
    upset_matrix = np.zeros((len(regions), len(rounds)))
    
    for i, region in enumerate(regions):
        for j, round_name in enumerate(rounds):
            if round_name == 'elite_8':
                upset_matrix[i, j] = 1 if results['bracket'][round_name][region]['upset'] else 0
            else:
                # Count upsets in this region for this round
                round_results = results['bracket'][round_name][region]
                upset_count = sum(1 for match in round_results if match['upset'])
                total_matches = len(round_results)
                upset_matrix[i, j] = upset_count / total_matches
    
    # Plot heatmap
    sns.heatmap(upset_matrix, annot=True, cmap='YlOrRd', 
                xticklabels=['First Round', 'Second Round', 'Sweet 16', 'Elite 8'],
                yticklabels=regions, fmt='.2f')
    plt.title('Upset Probability by Region and Round')
    plt.tight_layout()
    plt.savefig('upset_heatmap.png')
    plt.close()
    
    # Visualize the bracket structure
    self._visualize_bracket_structure(results)
    
    print("Visualizations saved to disk.")