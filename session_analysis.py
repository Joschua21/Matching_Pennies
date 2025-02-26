import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Global constants
DEFAULT_WINDOW_SIZE = 25  # Default window size for moving averages

plt.ioff()  # Turn off interactive mode

def load_data(file_path='Z:/delab/matchingpennies/matchingpennies_datatable.parquet'):
    """
    Load the matching pennies data from a parquet file and filter out ignored entries.

    Parameters:
    -----------
    file_path : str
        Path to the parquet file

    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    # Load the parquet file
    df = pd.read_parquet(file_path)

    # Filter out ignored entries
    df_filtered = df[df['ignore'] == 0]

    return df_filtered


def plot_choice_history(df, subject_id, session_id):
    """
    Plot the choice history for a specific subject and session.
    Empty circles indicate rewarded trials (reward=1), no markers for unrewarded trials.
    Gaps in the plot indicate missed trials (choice='M').
    """
    # Filter data for the specific subject and session
    subject_data = df[(df['subjid'] == subject_id) & (df['sessid'] == session_id)]

    if subject_data.empty:
        print(f"No data found for subject {subject_id} and session {session_id}")
        return

    # Extract choices and rewards
    choices = subject_data['choice'].values
    rewards = subject_data['reward'].values
    trials = np.arange(1, len(choices) + 1)

    # Create the figure
    fig = plt.figure(figsize=(12, 6))

    # Plot the choices
    for i, choice in enumerate(choices):
        if choice == 'L':
            plt.plot([i + 1, i + 1], [0, 1], 'r-', linewidth=1.5)
            if rewards[i] == 1:
                plt.plot(i + 1, 1, 'ro', markersize=8, fillstyle='none')
        elif choice == 'R':
            plt.plot([i + 1, i + 1], [0, -1], 'b-', linewidth=1.5)
            if rewards[i] == 1:
                plt.plot(i + 1, -1, 'bo', markersize=8, fillstyle='none')
        # 'M' choices result in a gap (no line)

    # Add the middle line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)

    # Set the y-axis limits and labels
    plt.ylim(-1.5, 1.5)
    plt.yticks([-1, 0, 1], ['Right', '', 'Left'])

    # Set the x-axis and title
    plt.xlabel('Trial Number')
    plt.title(f'Choice History for Subject {subject_id}, Session {session_id}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_reward_rate(df, subject_id, session_id, use_window_size=False):
    """
    Plot the reward rate over trials for a specific subject and session.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the matching pennies data
    subject_id : str
        Subject ID (e.g., 'JOA-M-0022')
    session_id : float
        Session ID (e.g., 456.0)
    use_window_size : bool
        If True, use DEFAULT_WINDOW_SIZE, otherwise use 10% of number of trials
    """
    # Filter data for the specific subject and session
    subject_data = df[(df['subjid'] == subject_id) & (df['sessid'] == session_id)]

    if subject_data.empty:
        print(f"No data found for subject {subject_id} and session {session_id}")
        return

    # Extract rewards
    rewards = subject_data['reward'].values
    trials = np.arange(1, len(rewards) + 1)

    # Determine window size
    if use_window_size:
        window_size = DEFAULT_WINDOW_SIZE
    else:
        window_size = max(int(len(trials) * 0.1), 1)  # Ensure at least 1

    # Calculate average reward rate for the entire session
    overall_rate = np.mean(rewards)

    # Calculate moving average
    reward_rates = []
    for i in range(len(trials)):
        if i < window_size:
            # For early trials, use a mix of actual data and overall average
            available_data = rewards[:i + 1]
            missing_data_weight = (window_size - len(available_data)) / window_size
            rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
        else:
            # Use the window of trials
            rate = np.mean(rewards[i - window_size + 1:i + 1])
        reward_rates.append(rate)

    # Create the figure
    fig = plt.figure(figsize=(12, 6))

    # Plot the reward rate
    plt.plot(trials, reward_rates, 'g-', linewidth=2)

    # Add a horizontal line at 0.5
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)

    # Set the y-axis limits and labels
    plt.ylim(0, 1)
    plt.ylabel('Reward Rate')

    # Set the x-axis and title
    plt.xlabel('Trial Number')
    plt.title(f'Reward Rate for Subject {subject_id}, Session {session_id} (Window Size: {window_size})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_prob_left(df, subject_id, session_id, use_window_size=False):
    """
    Plot the probability of choosing 'Left' over trials for a specific subject and session.
    Gaps in the plot indicate missed trials (choice='M').
    """
    # Filter data for the specific subject and session
    subject_data = df[(df['subjid'] == subject_id) & (df['sessid'] == session_id)]

    if subject_data.empty:
        print(f"No data found for subject {subject_id} and session {session_id}")
        return

    # Extract choices
    choices = subject_data['choice'].values

    # Convert choices to binary (1 for 'L', 0 for 'R', NaN for 'M')
    binary_choices = np.array([1 if c == 'L' else (0 if c == 'R' else np.nan) for c in choices])

    # Get valid trials (excluding 'M' choices)
    valid_trials = ~np.isnan(binary_choices)
    valid_binary = binary_choices[valid_trials]
    valid_indices = np.where(valid_trials)[0] + 1  # +1 for 1-based trial numbering

    # Determine window size
    if use_window_size:
        window_size = DEFAULT_WINDOW_SIZE
    else:
        window_size = max(int(np.sum(valid_trials) * 0.1), 1)  # Ensure at least 1

    # Calculate average left probability for the entire session
    overall_prob = np.mean(valid_binary)

    # Calculate moving average with weighted initialization
    left_probs = []
    all_probs = np.full(len(choices), np.nan)  # Initialize all trials with NaN
    
    for i in range(len(choices)):
        if choices[i] == 'M':
            continue  # Skip missed trials
            
        if i < window_size:
            # For early trials, use a mix of actual data and overall average
            available_data = binary_choices[:i + 1]
            valid_data = available_data[~np.isnan(available_data)]
            if len(valid_data) > 0:
                missing_data_weight = (window_size - len(valid_data)) / window_size
                prob = (np.sum(valid_data) + missing_data_weight * window_size * overall_prob) / window_size
                all_probs[i] = prob
        else:
            # Get window of valid trials before current trial
            window_choices = binary_choices[i - window_size + 1:i + 1]
            valid_window = window_choices[~np.isnan(window_choices)]
            if len(valid_window) > 0:
                prob = np.mean(valid_window)
                all_probs[i] = prob
    
    # Create the figure and plot
    fig = plt.figure(figsize=(12, 6))
    
    # Plot with gaps (NaN values will create gaps in the line)
    trials = np.arange(1, len(choices) + 1)
    plt.plot(trials, all_probs, 'b-', linewidth=2, label='Left Choice Probability')
    
    # Add a horizontal line at 0.5 (random chance)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (0.5)')
    
    # Set the y-axis limits and labels
    plt.ylim(0, 1)
    plt.ylabel('Probability of Left')

    # Set the x-axis and title
    plt.xlabel('Trial Number')
    plt.title(
        f'Probability of Left Choice for Subject {subject_id}, Session {session_id} (Window Size: {window_size})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return fig


def plot_prediction_history(df, subject_id, session_id):
    """
    Plot the computer prediction history for a specific subject and session.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the matching pennies data
    subject_id : str
        Subject ID (e.g., 'JOA-M-0022')
    session_id : float
        Session ID (e.g., 456.0)
    """
    # Filter data for the specific subject and session
    subject_data = df[(df['subjid'] == subject_id) & (df['sessid'] == session_id)]

    if subject_data.empty:
        print(f"No data found for subject {subject_id} and session {session_id}")
        return

    # Extract predictions
    predictions = subject_data['comp_prediction'].values
    trials = np.arange(1, len(predictions) + 1)

    # Create the figure
    fig = plt.figure(figsize=(12, 6))

    # Plot the predictions
    for i, pred in enumerate(predictions):
        if pred == 'L':
            plt.plot([i + 1, i + 1], [0, 1], 'r-', linewidth=1.5)
        elif pred == 'R':
            plt.plot([i + 1, i + 1], [0, -1], 'b-', linewidth=1.5)

    # Add the middle line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)

    # Set the y-axis limits and labels
    plt.ylim(-1.5, 1.5)
    plt.yticks([-1, 0, 1], ['Right', '', 'Left'])

    # Set the x-axis and title
    plt.xlabel('Trial Number')
    plt.title(f'Computer Prediction History for Subject {subject_id}, Session {session_id}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_computer_confidence(df, subject_id, session_id):
    """
    Plot the computer confidence based on p-values for a specific subject and session.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the matching pennies data
    subject_id : str
        Subject ID (e.g., 'JOA-M-0022')
    session_id : float
        Session ID (e.g., 456.0)
    """
    # Filter data for the specific subject and session
    subject_data = df[(df['subjid'] == subject_id) & (df['sessid'] == session_id)]

    if subject_data.empty:
        print(f"No data found for subject {subject_id} and session {session_id}")
        return

    # Extract p-values
    p_values = subject_data['min_pvalue'].values
    trials = np.arange(1, len(p_values) + 1)

    # Cap very small p-values at 10^-12 to avoid infinite confidence
    min_p_value = 1e-12
    p_values = np.maximum(p_values, min_p_value)

    # Calculate confidence as -log10(p_value)
    confidence = -np.log10(p_values)

    # Create the figure
    fig = plt.figure(figsize=(12, 6))

    # Plot the confidence
    plt.plot(trials, confidence, 'g-', linewidth=2)

    # Add a horizontal line at significance level 0.05
    significance_level = -np.log10(0.05)
    plt.axhline(y=significance_level, color='r', linestyle='--', alpha=0.5)

    # Set the y-axis and labels
    plt.ylabel('Computer Confidence (-log10(p))')

    # Set the x-axis and title
    plt.xlabel('Trial Number')
    plt.title(f'Computer Confidence for Subject {subject_id}, Session {session_id}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def analyze_patterns(df, subject_id, session_id):
    """
    Analyze and plot the frequency of 3-choice patterns for a specific subject and session.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the matching pennies data
    subject_id : str
        Subject ID (e.g., 'JOA-M-0022')
    session_id : float
        Session ID (e.g., 456.0)
    """
    # Filter data for the specific subject and session
    subject_data = df[(df['subjid'] == subject_id) & (df['sessid'] == session_id)]

    if subject_data.empty:
        print(f"No data found for subject {subject_id} and session {session_id}")
        return

    # Extract choices, filtering out 'M'
    all_choices = subject_data['choice'].values
    choices = [c for c in all_choices if c in ('L', 'R')]

    if len(choices) < 3:
        print(f"Not enough valid choices (need at least 3) for subject {subject_id} and session {session_id}")
        return

    # Initialize pattern dictionary for all possible 3-choice patterns
    patterns = {
        'LLL': 0, 'LLR': 0, 'LRL': 0, 'LRR': 0,
        'RLL': 0, 'RLR': 0, 'RRL': 0, 'RRR': 0
    }

    # Count patterns
    for i in range(len(choices) - 2):
        pattern = choices[i] + choices[i + 1] + choices[i + 2]
        if pattern in patterns:
            patterns[pattern] += 1

    # Calculate cumulative frequencies
    pattern_counts = {k: 0 for k in patterns.keys()}
    cumulative_counts = {k: [] for k in patterns.keys()}
    x_values = list(range(3, len(choices) + 1))

    for i in range(len(choices) - 2):
        pattern = choices[i] + choices[i + 1] + choices[i + 2]
        if pattern in pattern_counts:
            pattern_counts[pattern] += 1

        # Update cumulative counts for all patterns
        for p in patterns.keys():
            cumulative_counts[p].append(pattern_counts.get(p, 0))

    # Create the figure
    fig = plt.figure(figsize=(12, 8))

    # Plot cumulative frequencies for each pattern
    for pattern, counts in cumulative_counts.items():
        plt.plot(x_values, counts, label=pattern)

    # Set the y-axis and labels
    plt.ylabel('Cumulative Frequency')

    # Set the x-axis and title
    plt.xlabel('Trial Number')
    plt.title(f'Cumulative Pattern Frequencies for Subject {subject_id}, Session {session_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return fig


def analyze_subject(df, subject_id, session_id, save_path=None, show_plots=True):
    """
    Run all analyses for a specific subject and session, and optionally save the figures.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the matching pennies data
    subject_id : str
        Subject ID (e.g., 'JOA-M-0022')
    session_id : float
        Session ID (e.g., 456.0)
    save_path : str, optional
        Path to save the figures. If None, figures are not saved.
    show_plots : bool, default=True
        Whether to display the plots
    """
    # Create a directory for saving figures if specified
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Run all analyses
    analyses = {
        'choice_history': plot_choice_history(df, subject_id, session_id),
        'reward_rate': plot_reward_rate(df, subject_id, session_id),
        'prob_left': plot_prob_left(df, subject_id, session_id),
        'prediction_history': plot_prediction_history(df, subject_id, session_id),
        'computer_confidence': plot_computer_confidence(df, subject_id, session_id),
        'patterns': analyze_patterns(df, subject_id, session_id)
    }

    # Save figures if requested
    if save_path:
        for name, fig in analyses.items():
            if fig:
                fig.savefig(save_dir / f"{subject_id}_{session_id}_{name}.png", dpi=300, bbox_inches='tight')

    # Show or close figures
    if not show_plots:
        for fig in analyses.values():
            if fig:
                plt.close(fig)

    return analyses