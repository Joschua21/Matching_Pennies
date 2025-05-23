import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import glob
import warnings
import pandas as pd
import scipy.stats
from scipy.stats import chi2_contingency, fisher_exact

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global parameters
base_dir = "/Volumes/ogma/delab/lab-members/joanna/photometry/preprocess"
output_dir = "/Volumes/ogma/delab/lab-members/joschua/photometry_analysis"
sampling_rate = 120
pre_cue_time = 3
post_cue_time = 5
pre_cue_samples = int(pre_cue_time * sampling_rate)
post_cue_samples = int(post_cue_time * sampling_rate)
total_window_samples = pre_cue_samples + post_cue_samples

PARQUET_PATH = "/Volumes/ogma/delab/matchingpennies/matchingpennies_datatable.parquet"
CODE_VERSION = "1.1.1"  # Increment this when making analysis changes --> will force recomputation of all data
_SESSION_CACHE = {}


def clear_memory():
    #used to clear the session cache to reduce working memory utilization
    global _SESSION_CACHE
    _SESSION_CACHE = {}
    import gc
    gc.collect()

def calculate_sem(data, axis=0):
    """Calculate Standard Error of Mean (SEM)"""
    std = np.std(data, axis=axis)
    n = data.shape[axis]
    return std / np.sqrt(n)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory {directory}")


def get_output_path(subject_id, session_date):
    subject_dir = os.path.join(output_dir, subject_id)
    session_dir = os.path.join(subject_dir, session_date)

    ensure_directory_exists(subject_dir)
    ensure_directory_exists(session_dir)

    return session_dir


def load_filtered_behavior_data(protocol_filter="MatchingPennies", subject_id=None, ignore=0, behavior_df=None):
    """
    Load behavioral data, filtering for a specific protocol, subject, and ignore status.
    Returns the filtered DataFrame for reuse across functions.
    
    Parameters:
    -----------
    protocol_filter : str, optional
        String that must be contained in protocol name
    subject_id : str, optional
        If provided, filter for this specific subject
    ignore : int or None, optional
        If provided, filter for this ignore value (default=0)
    behavior_df : pandas.DataFrame, optional
        Pre-loaded dataframe to filter instead of loading from parquet
        
    Returns:
    --------
    pandas.DataFrame: Filtered behavioral data
    """
    try:
        if behavior_df is None:
            # Load from parquet if no dataframe provided
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
        else:
            # Use the provided dataframe
            df = behavior_df.copy()
            if 'date' in df.columns and df['date'].dtype != str:
                df['date'] = df['date'].astype(str)
        
        # Create a mask for protocol filtering
        mask = df['protocol'].str.contains(protocol_filter, na=False) 
        
        # Add ignore filter if provided
        if ignore is not None:
            mask &= (df["ignore"] == ignore)
            
        # Add subject filter if provided
        if subject_id is not None:
            mask &= (df["subjid"] == subject_id)
            
        # Apply all filters at once
        filtered_df = df[mask]
        
        # Print summary info
        source = 'provided dataframe' if behavior_df is not None else 'parquet file'
        filter_info = f"protocol='{protocol_filter}', ignore={ignore}"
        if subject_id:
            filter_info += f", subject='{subject_id}'"
            
        print(f"Using {source}: {len(filtered_df)} trials matching {filter_info}")
        return filtered_df
    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        return None

def load_behavior_data(subject_id, session_date, behavior_df=None):
    """Load behavioral data for a specific subject and session date"""
    try:
        if behavior_df is None:
            # Load the parquet file with pyarrow
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
        else:
            # Use the provided dataframe
            df = behavior_df.copy()
            if df['date'].dtype != str:
                df['date'] = df['date'].astype(str)
                
        session_data = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
        
        if session_data.empty:
            print(f"No behavioral data found for {subject_id} on {session_date}")

        lick_timestamps = session_data["LickTimestamp"].values
        sound_timestamps = session_data["SoundTimestamp"].values
        corrections = (lick_timestamps - sound_timestamps) * sampling_rate

        return {
            "choice": session_data["choice"].tolist(),  # List of 'R' or 'L'
            "lick_timestamps": lick_timestamps,  # List of lick times
            "soundtimestamps": sound_timestamps,
            "corrections": corrections,
            "reward": session_data["reward"].tolist(),
        }

    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        return None
    

def find_pkl_file(directory):
    """Find the first .pkl file in the given directory"""
    pkl_files = glob.glob(os.path.join(directory, "*.pkl"))
    return pkl_files[0] if pkl_files else None


def check_saved_results(subject_id, session_date):
    """Checks if calculation has already be done and results are stored in a .pkl file. If Code Version changed, recomputes and overwrites results file"""
    session_dir = get_output_path(subject_id, session_date)
    results_file = os.path.join(session_dir, "analysis_results.pkl")

    if os.path.exists(results_file):
        try:
            with open(results_file, "rb") as f:
                result = pickle.load(f)
            # Check if saved result has version info and matches current version
            if 'code_version' not in result or result['code_version'] != CODE_VERSION:
                print(f"Saved results for {subject_id}/{session_date} are from different code version. Recomputing...")
                return None
            print(f"Loaded saved results for {subject_id}/{session_date}")
            return result
        except Exception as e:
            print(f"Error loading saved results: {e}")

    return None


def save_results(result, subject_id, session_date):
    if not result:
        return

    session_dir = get_output_path(subject_id, session_date)
    results_file = os.path.join(session_dir, "analysis_results.pkl")

    try:
        with open(results_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved saved results for {subject_id}/{session_date} to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def save_figure(fig, subject_id, session_date, fig_name="figure"):
    """Save figure to file with error handling"""
    if fig is None:
        print("Warning: No figure to save")
        return

    session_dir = get_output_path(subject_id, session_date)
    fig_file = os.path.join(session_dir, f"{fig_name}.png")

    try:
        fig.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure for {subject_id}/{session_date} to {fig_file}")
    except Exception as e:
        print(f"Error saving figure: {e}")


def process_session(subject_id, session_date, force_recompute=False, use_global_cache=True, behavior_df=None, z_score=True):
    """Process a single session for a given subject"""


    cache_key = f"{subject_id}/{session_date}" + ("" if z_score else "/raw")
    if use_global_cache and cache_key in _SESSION_CACHE and not force_recompute:
        return _SESSION_CACHE[cache_key]
    
    # Then check saved results
    if not force_recompute:
        session_dir = get_output_path(subject_id, session_date)
        results_file = os.path.join(session_dir, "analysis_results" + ("" if z_score else "_raw") + ".pkl")
        
        if os.path.exists(results_file):
            try:
                with open(results_file, "rb") as f:
                    result = pickle.load(f)
                # Check if saved result has version info and matches current version
                if 'code_version' not in result or result['code_version'] != CODE_VERSION:
                    print(f"Saved results for {subject_id}/{session_date} are from different code version. Recomputing...")
                else:
                    print(f"Loaded saved {'z-scored' if z_score else 'raw'} results for {subject_id}/{session_date}")
                    if use_global_cache:
                        _SESSION_CACHE[cache_key] = result
                    return result
            except Exception as e:
                print(f"Error loading saved results: {e}")

    full_dir = os.path.join(base_dir, subject_id, session_date)
    deltaff_file = os.path.join(full_dir, "deltaff.npy")
    pkl_file = find_pkl_file(full_dir)

    if not pkl_file or not os.path.exists(deltaff_file):
        print(f"Missing files for {subject_id}/{session_date}")
        return None

    try:
        deltaff_data = np.load(deltaff_file)
        print(f"{subject_id}/{session_date}: Loaded deltaff data with shape: {deltaff_data.shape}")
        
        # Only apply z-scoring if requested
        if z_score:
            deltaff_mean = np.mean(deltaff_data)
            detlaff_std = np.std(deltaff_data)
            deltaff_data = (deltaff_data - deltaff_mean) / detlaff_std
            print(f"{subject_id}/{session_date}: Applied z-score normalization")
        else:
            print(f"{subject_id}/{session_date}: Using raw (non-z-scored) data")

        with open(pkl_file, "rb") as f:
            pkl_data = pickle.load(f)
        print(f"{subject_id}/{session_date}: Loaded pkl data: {os.path.basename(pkl_file)}")

        # Extract pulse indices
        pulse_indices = pkl_data["pulse_inds_1"]
        num_trials = len(pulse_indices)
        print(f"{subject_id}/{session_date}: Found {num_trials} pulse indices")

        behavior_data = load_behavior_data(subject_id, session_date, behavior_df)
        if behavior_data is None:
            print(f"Warning: No behavioral data found for {subject_id} on {session_date}")
            return None

        corrections = np.array(behavior_data['corrections'])

        if len(pulse_indices) != len(corrections):
            print(
                f"Mismatch between pulse indices and corrections for {subject_id}/{session_date}. Skipping correction")
            corrected_indices = pulse_indices
        else:
            corrected_indices = pulse_indices + corrections.astype(int)

        # Epoch the data
        epoched_data = np.zeros((num_trials, total_window_samples))
        valid_trials = []
        choice_values = np.array(behavior_data["choice"])

        for i, index in enumerate(corrected_indices):
            if index >= pre_cue_samples and index + post_cue_samples < len(deltaff_data):
                start_idx = index - pre_cue_samples
                end_idx = index + post_cue_samples
                epoched_data[i] = deltaff_data[start_idx:end_idx]
                valid_trials.append(i)
            else:
                epoched_data[i] = np.nan

        # Count actual M trials instead of using difference
        non_m_trials = []
        m_trial_count = 0
        for i in valid_trials:
            if i < len(choice_values):
                if choice_values[i] == 'M':
                    m_trial_count += 1
                else:
                    non_m_trials.append(i)

        plotting_data = epoched_data[non_m_trials]
        all_epoched_data = epoched_data[valid_trials]

        print(f"{subject_id}/{session_date}: Successfully epoched {len(valid_trials)} valid trials")
        print(f"{subject_id}/{session_date}: {m_trial_count} missed trials will be excluded from statistics and plots")

        # Calculate statistics
        time_axis = np.linspace(-pre_cue_time, post_cue_time, total_window_samples, endpoint=False)
        trial_average = np.mean(plotting_data, axis=0)
        trial_sem = calculate_sem(plotting_data, axis=0)

        reward_outcomes = np.array(behavior_data["reward"])[valid_trials]
        choices = np.array(behavior_data["choice"])[valid_trials]

        result = {
            'code_version': CODE_VERSION,  # Add version to saved results
            'subject_id': subject_id,
            'session_date': session_date,
            'epoched_data': all_epoched_data,
            'time_axis': time_axis,
            'trial_average': trial_average,
            'trial_sem': trial_sem,
            'valid_trials': valid_trials,
            'non_m_trials': non_m_trials,
            'num_trials': num_trials,
            'behavioral_data': behavior_data,
            'reward_outcomes': reward_outcomes,
            'choices': choices,
            'plotting_data': plotting_data,
            'z_scored': z_score  # Store whether this data was z-scored
        }

        # Save results with appropriate filename suffix based on z_score parameter
        session_dir = get_output_path(subject_id, session_date)
        results_file = os.path.join(session_dir, f"analysis_results{'_raw' if not z_score else ''}.pkl")
        
        try:
            with open(results_file, "wb") as f:
                pickle.dump(result, f)
            print(f"Saved {'raw' if not z_score else ''} results for {subject_id}/{session_date} to {results_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
            
        # Store in memory cache
        if use_global_cache:
            _SESSION_CACHE[cache_key] = result
            
        return result

    except Exception as e:
        print(f"Error processing session {subject_id}/{session_date}: {e}")
        return None

def plot_session_results(analysis_result, show_heatmap=False, win_loss=False, save_fig=True):
    """Plot the results for a single session with an optional heatmap of individual trials."""
    if not analysis_result:
        print("No results to plot")
        return

    # Set up subplots: 1x2 if heatmap is enabled, else just 1 plot
    fig, axes = plt.subplots(1, 2 if show_heatmap else 1, figsize=(16 if show_heatmap else 12, 7))

    # Ensure `axes` is always iterable
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # First subplot: Trial Average ± Standard Deviation
    ax1 = axes[0]
    if win_loss and isinstance(analysis_result, dict) and "reward_outcomes" in analysis_result:
        non_m_indices = np.array([i for i, idx in enumerate(analysis_result["valid_trials"])
                                  if idx in analysis_result["non_m_trials"]])
        filtered_reward_outcomes = analysis_result["reward_outcomes"][non_m_indices]

        # Split trials by reward outcome
        rewarded_trials = analysis_result["plotting_data"][filtered_reward_outcomes == 1]
        unrewarded_trials = analysis_result["plotting_data"][filtered_reward_outcomes == 0]

        # Plot rewarded trials
        if len(rewarded_trials) > 0:
            rewarded_avg = np.mean(rewarded_trials, axis=0)
            rewarded_sem = calculate_sem(rewarded_trials, axis=0)
            ax1.fill_between(analysis_result['time_axis'], rewarded_avg - rewarded_sem,
                             rewarded_avg + rewarded_sem, color='lightgreen', alpha=0.4, label='Rewarded ± SEM')
            ax1.plot(analysis_result['time_axis'], rewarded_avg, color='green', linewidth=2.5, label='Rewarded Avg')

        # Plot unrewarded trials
        if len(unrewarded_trials) > 0:
            unrewarded_avg = np.mean(unrewarded_trials, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)
            ax1.fill_between(analysis_result['time_axis'], unrewarded_avg - unrewarded_sem,
                             unrewarded_avg + unrewarded_sem, color='lightsalmon', alpha=0.4, label='Unrewarded ± SEM')
            ax1.plot(analysis_result['time_axis'], unrewarded_avg, color='darkorange', linewidth=2.5,
                     label='Unrewarded Avg')

    else:  # Default behavior
        trial_sem = calculate_sem(analysis_result['plotting_data'], axis=0)
        ax1.fill_between(analysis_result['time_axis'],
                         analysis_result['trial_average'] - trial_sem,
                         analysis_result['trial_average'] + trial_sem,
                         color='lightgreen', alpha=0.4, label='Mean ± SEM')

        ax1.plot(analysis_result['time_axis'], analysis_result['trial_average'],
                 color='darkgreen', linewidth=2.5, label='Trial Average')

    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax1.set_xlabel('Time (s)', fontsize=16)
    ax1.set_ylabel('ΔF/F', fontsize=16)
    ax1.set_title(f'Photometry Response: {analysis_result["subject_id"]} - {analysis_result["session_date"]}',
                  fontsize=20)
    ax1.set_xlim([-pre_cue_time, post_cue_time])
    ax1.legend(loc='upper right', fontsize=16)

    # Add statistics text box
    stats_text = (f"Trials: {len(analysis_result['non_m_trials'])} (excluding missed trials)\n"
                  f"Peak: {np.max(analysis_result['trial_average']):.4f}\n"
                  f"Baseline: {np.mean(analysis_result['trial_average'][:pre_cue_samples]):.4f}")
    ax1.text(-pre_cue_time + 0.2, np.max(analysis_result['trial_average']) * 0.9, stats_text,
             bbox=dict(facecolor='white', alpha=0.7), fontsize=16)

    # Second subplot: Heatmap (if enabled)
    if show_heatmap:
        ax2 = axes[1]
        trial_data = analysis_result['plotting_data']
        im = ax2.imshow(trial_data,
                        aspect='auto',
                        extent=[-pre_cue_time, post_cue_time, 0, len(trial_data)],
                        cmap='viridis',
                        interpolation='nearest')

        ax2.set_xlabel('Time (s)', fontsize=16)
        ax2.set_ylabel('Trial Number', fontsize=16)
        ax2.set_title('Trial-by-Trial Heatmap', fontsize=20)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)

        # Add colorbar
        plt.colorbar(im, ax=ax2, label='ΔF/F')

    plt.tight_layout()

    if save_fig:
        save_figure(fig, analysis_result["subject_id"], analysis_result["session_date"],
                    f"session_results{'_heatmap' if show_heatmap else ''}{'_winloss' if win_loss else ''}")

    plt.show()
    return analysis_result


def check_saved_pooled_results(subject_id, win_loss=False):
    subject_dir = os.path.join(output_dir, subject_id)
    filename = f"pooled_results{'_winloss' if win_loss else ''}.pkl"
    pooled_file = os.path.join(subject_dir, filename)

    if os.path.exists(pooled_file):
        try:
            with open(pooled_file, "rb") as f:
                results = pickle.load(f)
            # Check version
            if 'code_version' not in results or results['code_version'] != CODE_VERSION:
                print(f"Saved pooled results for {subject_id} are from different code version. Recomputing...")
                return None
            print(f"Loaded saved pooled results for {subject_id}")
            return results
        except Exception as e:
            print(f"Error loading saved pooled results: {e}")
    return None


def save_pooled_results(result, subject_id, win_loss=False):
    """Save pooled analysis results to file"""
    if not result:
        return

    subject_dir = os.path.join(output_dir, subject_id)
    ensure_directory_exists(subject_dir)

    filename = f"pooled_results{'_winloss' if win_loss else ''}.pkl"
    pooled_file = os.path.join(subject_dir, filename)

    try:
        with open(pooled_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved pooled results to {pooled_file}")
    except Exception as e:
        print(f"Error saving pooled results: {e}")


def analyze_pooled_data(subject_id, win_loss=False, force_recompute=False, fig=None, show_session_traces=False, behavior_df=None, suppress_plotting=False):
    """Analyze and visualize pooled data for a subject"""
    # Create figure if not provided and not suppressing plotting
    plt.style.use('default')
    
    if fig is None and not suppress_plotting:
        fig = plt.figure(figsize=(12, 7))

    # Check for saved pooled results
    if not force_recompute:
        saved_results = check_saved_pooled_results(subject_id, win_loss)
        if saved_results is not None:
            print(f"Using saved results for {subject_id}")
            
            # Only create visualization if not suppressing plotting
            if not suppress_plotting:
                # This way, we can control show_session_traces without recomputing
                time_axis = saved_results['time_axis']
                
                # Create a new figure for the visualization
                plt.figure(figsize=(12, 7))
                
                # Optionally plot session traces
                if show_session_traces and 'session_averages' in saved_results:
                    blue_colors = plt.cm.Blues(np.linspace(0.3, 1, len(saved_results['session_dates'])))
                    for idx, (session_date, session_avg) in enumerate(zip(saved_results['session_dates'], saved_results['session_averages'])):
                        plt.plot(time_axis, session_avg,
                                alpha=0.6, linewidth=1, linestyle='-',
                                color=blue_colors[idx],
                                label=f"Session {session_date}")
                
                # Plot the main data (win/loss or average)
                if win_loss:
                    if saved_results['rewarded_avg'] is not None:
                        plt.fill_between(time_axis, 
                                       saved_results['rewarded_avg'] - saved_results['rewarded_sem'],  
                                       saved_results['rewarded_avg'] + saved_results['rewarded_sem'],  
                                       color='lightgreen', alpha=0.4, label='Rewarded ± SEM')  
                        plt.plot(time_axis, saved_results['rewarded_avg'], 
                               color='green', linewidth=2.5, label='Rewarded Avg')

                    if saved_results['unrewarded_avg'] is not None:
                        plt.fill_between(time_axis, 
                                       saved_results['unrewarded_avg'] - saved_results['unrewarded_sem'],  
                                       saved_results['unrewarded_avg'] + saved_results['unrewarded_sem'],  
                                       color='lightsalmon', alpha=0.4, label='Unrewarded ± SEM')  
                        plt.plot(time_axis, saved_results['unrewarded_avg'], 
                               color='darkorange', linewidth=2.5, label='Unrewarded Avg')
                else:
                    plt.fill_between(time_axis,
                                   saved_results['pooled_average'] - saved_results['pooled_sem'],  
                                   saved_results['pooled_average'] + saved_results['pooled_sem'],  
                                   color='lightgreen', alpha=0.4,
                                   label='Mean ± SEM')  
                    plt.plot(time_axis, saved_results['pooled_average'], 
                           color='green', linewidth=2.5, label='Overall Avg')
                
                # Add vertical line at cue onset
                plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
                plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # Labels and formatting
                plt.xlabel('Time (s) after first lick', fontsize=24)
                plt.ylabel('z-ΔF/F', fontsize=24)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.title(f'Pooled Photometry Response: {subject_id} ({len(saved_results["session_dates"])} sessions)', fontsize=26)
                plt.xlim([-pre_cue_time, post_cue_time])
                
                # Fix the legend - limit session traces if too many
                if show_session_traces and 'session_dates' in saved_results and len(saved_results['session_dates']) > 5:
                    handles, labels = plt.gca().get_legend_handles_labels()
                    # Find how many session labels we have
                    session_handles = [h for h, l in zip(handles, labels) if l.startswith("Session")]
                    non_session_handles = [h for h, l in zip(handles, labels) if not l.startswith("Session")]
                    session_labels = [l for l in labels if l.startswith("Session")]
                    non_session_labels = [l for l in labels if not l.startswith("Session")]
                    
                    # Only keep the first 5 session labels
                    limited_session_handles = session_handles[:5]
                    limited_session_labels = session_labels[:5]

                    # Show limited legend
                    plt.legend(limited_session_handles + non_session_handles, 
                              limited_session_labels + non_session_labels, 
                              loc='upper right', fontsize=10)
                else:
                    plt.legend(loc='upper right', fontsize=10)
                    
                # Add grid explicitly before other elements
                plt.grid(True, alpha=0.3)  # Make sure this comes before tight_layout

                plt.tight_layout()
                # Save the figure with appropriate suffix
                trace_suffix = "_with_sessions" if show_session_traces else ""
                save_figure(plt.gcf(), subject_id, "pooled", f"pooled_results{trace_suffix}{'_winloss' if win_loss else ''}")
                
                plt.show()
                
            return saved_results

    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None

    # Process each session and collect results
    all_sessions = []
    all_plotting_data = []
    session_dates = []
    session_averages = []

    # Get matching pennies sessions from behavior dataframe if provided, otherwise load from parquet
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d)) and
                      os.path.exists(os.path.join(subject_dir, d, "deltaff.npy")) and
                      (d in matching_pennies_sessions or not matching_pennies_sessions)])
    
    num_sessions = len(sessions)
    blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))  # Create blue gradient

    for idx, session_date in enumerate(sessions):
        session_path = os.path.join(subject_dir, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            # Pass behavior_df to process_session to reuse data
            result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if result:
                if len(result['non_m_trials']) < 100:
                    print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(result['non_m_trials'])}).")
                    continue

                all_sessions.append(result)
                all_plotting_data.append(result['plotting_data'])
                session_dates.append(session_date)
                session_averages.append(result['trial_average'])

    if not all_sessions:
        print(f"No processed sessions found for subject {subject_id}")
        return None

    # Concatenate all trials from all sessions
    pooled_data = np.vstack(all_plotting_data)
    pooled_average = np.mean(pooled_data, axis=0)
    time_axis = all_sessions[0]['time_axis']

    rewarded_avg = None
    unrewarded_avg = None
    rewarded_data = np.array([])
    unrewarded_data = np.array([])

    if win_loss:
        rewarded_data = []
        unrewarded_data = []
        for session in all_sessions:
            non_m_indices = np.array(
                [i for i, idx in enumerate(session["valid_trials"]) if idx in session["non_m_trials"]])
            filtered_reward_outcomes = session["reward_outcomes"][non_m_indices]

            session_plot_data = session["plotting_data"]

            if len(session_plot_data) > 0:
                session_rewarded = session_plot_data[filtered_reward_outcomes == 1]
                session_unrewarded = session_plot_data[filtered_reward_outcomes == 0]

                if len(session_rewarded) > 0:
                    rewarded_data.append(session_rewarded)
                if len(session_unrewarded) > 0:
                    unrewarded_data.append(session_unrewarded)

        # Filter out empty arrays before stacking
        if rewarded_data:
            rewarded_data = np.vstack(rewarded_data)
        if unrewarded_data:
            unrewarded_data = np.vstack(unrewarded_data)

    # Only create plots if not suppressing visualization
    if not suppress_plotting:
        # Create the pooled plot
        plt.figure(figsize=(12, 7))
        
        # Plot reward outcomes or average data
        if win_loss:
            # Compute averages and std deviations
            if rewarded_data.size > 0:
                rewarded_avg = np.mean(rewarded_data, axis=0)
                rewarded_sem = calculate_sem(rewarded_data, axis=0)  
                plt.fill_between(time_axis, 
                                 rewarded_avg - rewarded_sem,  
                                 rewarded_avg + rewarded_sem,  
                                 color='lightgreen', alpha=0.4, label='Rewarded ± SEM')  
                plt.plot(time_axis, rewarded_avg, color='green', linewidth=2.5, label='Rewarded Avg')

            if unrewarded_data.size > 0:
                unrewarded_avg = np.mean(unrewarded_data, axis=0)
                unrewarded_sem = calculate_sem(unrewarded_data, axis=0) 
                plt.fill_between(time_axis, 
                                 unrewarded_avg - unrewarded_sem,  
                                 unrewarded_avg + unrewarded_sem,  
                                 color='lightsalmon', alpha=0.4, label='Unrewarded ± SEM')  
                plt.plot(time_axis, unrewarded_avg, color='darkorange', linewidth=2.5, label='Unrewarded Avg')
        else:
            pooled_sem = calculate_sem(pooled_data, axis=0)  
            plt.fill_between(time_axis,
                             pooled_average - pooled_sem,  
                             pooled_average + pooled_sem,  
                             color='lightgreen', alpha=0.4,
                             label='Mean ± SEM')  
            plt.plot(time_axis, pooled_average, color='green', linewidth=2.5, label='Overall Avg')

        # Add session traces if requested
        if show_session_traces:
            # Plot individual session averages with blue gradient
            for idx, (session_date, session_avg) in enumerate(zip(session_dates, session_averages)):
                plt.plot(time_axis, session_avg,
                         alpha=0.6, linewidth=1, linestyle='-',
                         color=blue_colors[idx],  # Use blue gradient
                         label=f"Session {session_date}")
                
        # Add a vertical line at the cue onset (time=0)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Labels and formatting
        plt.xlabel('Time (s) after first lick', fontsize=24)
        plt.ylabel('z-ΔF/F', fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(f'Pooled Photometry Response: {subject_id} ({len(all_sessions)} sessions)', fontsize=26)
        plt.xlim([-pre_cue_time, post_cue_time])

        # Limit legend items if too many sessions
        if len(all_sessions) > 5:
            handles, labels = plt.gca().get_legend_handles_labels()
            limited_handles = handles[:8]
            limited_labels = labels[:8]
            plt.legend(limited_handles, limited_labels, loc='upper right', fontsize=24)
        else:
            plt.legend(loc='upper right', fontsize=24)

        plt.tight_layout()

        # Save the figure
        trace_suffix = "_with_sessions" if show_session_traces else ""
        save_figure(plt.gcf(), subject_id, "pooled", f"pooled_results{trace_suffix}{'_winloss' if win_loss else ''}")

        plt.show()

    # Calculate SEMs for the return value
    rewarded_sem = calculate_sem(rewarded_data, axis=0) if rewarded_data.size > 0 else None
    unrewarded_sem = calculate_sem(unrewarded_data, axis=0) if unrewarded_data.size > 0 else None
    
    # Always prepare pooled result regardless of plotting
    pooled_result = {
        'code_version': CODE_VERSION,
        'subject_id': subject_id,
        'session_dates': session_dates,
        'pooled_data': pooled_data,
        'pooled_average': pooled_average,
        'pooled_sem': calculate_sem(pooled_data, axis=0),  
        'time_axis': time_axis,
        'total_trials': sum(len(session['non_m_trials']) for session in all_sessions),
        'session_averages': session_averages,
        'rewarded_avg': np.mean(rewarded_data, axis=0) if rewarded_data.size > 0 else None,
        'rewarded_sem': rewarded_sem,
        'unrewarded_avg': np.mean(unrewarded_data, axis=0) if unrewarded_data.size > 0 else None,
        'unrewarded_sem': unrewarded_sem
    }

    # Save pooled results
    save_pooled_results(pooled_result, subject_id, win_loss)


def analyze_all_subjects(win_loss=False, force_recompute=False):
    """Process all subjects and return results without saving"""
    # Find all subject directories
    all_results = {}
    all_subjects = []

    for subject_id in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        print(f"\nProcessing subject: {subject_id}")

        # Check if the subject has any valid sessions
        has_valid_sessions = False
        for session_date in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session_date)
            if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
                has_valid_sessions = True
                break

        if has_valid_sessions:
            all_subjects.append(subject_id)
            # Pass win_loss parameter to analyze_pooled_data
            pooled_result = analyze_pooled_data(subject_id, win_loss=win_loss)
            if pooled_result:
                all_results[subject_id] = pooled_result

    print(f"\nProcessed {len(all_results)} subjects with data")
    return all_results, all_subjects


def select_and_visualize(show_heatmap=False, win_loss=False, force_recompute=False):
    """Interactive function to select and visualize data"""
    # Get list of all subjects
    subjects = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))]

    if not subjects:
        print(f"No subject directories found in {base_dir}")
        return

    print("Available subjects:")
    for i, subj in enumerate(subjects):
        print(f"{i + 1}. {subj}")

    # Select a subject
    try:
        subj_idx = int(input("Enter subject number: ")) - 1
        if subj_idx < 0 or subj_idx >= len(subjects):
            print("Invalid subject number")
            return

        selected_subject = subjects[subj_idx]
    except ValueError:
        print("Please enter a valid number")
        return

    # Ask for win/loss analysis preference
    win_loss_input = input("Show win/loss analysis? (y/n): ").lower()
    win_loss = win_loss_input.startswith('y')

    # Get sessions for this subject
    subject_dir = os.path.join(base_dir, selected_subject)
    sessions = [d for d in os.listdir(subject_dir)
                if os.path.isdir(os.path.join(subject_dir, d)) and
                os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))]

    if not sessions:
        print(f"No valid sessions found for subject {selected_subject}")
        return

    print(f"\nAvailable sessions for {selected_subject}:")
    for i, sess in enumerate(sessions):
        print(f"{i + 1}. {sess}")
    print(f"{len(sessions) + 1}. [POOLED] All sessions")

    # Select a session or pooled analysis
    try:
        sess_idx = int(input("Enter session number or select pooled analysis: ")) - 1
        if sess_idx < 0 or sess_idx > len(sessions):
            print("Invalid selection")
            return

        if sess_idx == len(sessions):  # Pooled analysis selected
            print(f"\nGenerating pooled analysis for {selected_subject}...")
            analyze_pooled_data(selected_subject, win_loss=win_loss)
        else:  # Single session selected
            selected_session = sessions[sess_idx]
            print(f"\nAnalyzing session {selected_subject}/{selected_session}...")
            result = process_session(selected_subject, selected_session)
            if result:
                plot_session_results(result, show_heatmap=show_heatmap, win_loss=win_loss)
            else:
                print("Failed to process session")

    except ValueError:
        print("Please enter a valid number")
        return


def analyze_specific_session(subject_id, session_date, show_heatmap=False, win_loss=False):
    """Analyze and visualize a specific session"""
    print(f"Analyzing session {subject_id}/{session_date}...")
    analysis_result = process_session(subject_id, session_date)
    if analysis_result:
        return plot_session_results(analysis_result, show_heatmap=show_heatmap, win_loss=win_loss)
    return None


def pooled_results(subject_id="All", win_loss=False, force_recompute=False, show_session_traces=False, behavior_df=None, specific_subjects=None):
    """Analyze and visualize pooled results for a subject or across all subjects"""
    
    # Handle all-subject analysis
    if subject_id == "All":
        if specific_subjects is None:
            # Default list of subjects if not specified
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Store individual subject averages
        all_subject_averages = []
        all_subject_sem = []
        time_axis = None
        total_sessions = 0
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for cross-subject averaging...")
            result = analyze_pooled_data(subj, win_loss=win_loss, force_recompute=force_recompute, 
                               show_session_traces=False, behavior_df=behavior_df, 
                               suppress_plotting=True) 
            
            if result:
                if time_axis is None:
                    time_axis = result['time_axis']
                
                if win_loss:
                    # For win-loss analysis, store both rewarded and unrewarded data
                    if result['rewarded_avg'] is not None:
                        all_subject_averages.append({"type": "rewarded", "data": result['rewarded_avg']})
                        all_subject_sem.append({"type": "rewarded", "data": result['rewarded_sem']})
                    
                    if result['unrewarded_avg'] is not None:
                        all_subject_averages.append({"type": "unrewarded", "data": result['unrewarded_avg']})
                        all_subject_sem.append({"type": "unrewarded", "data": result['unrewarded_sem']})
                else:
                    # For regular analysis, just store the pooled average
                    all_subject_averages.append({"type": "average", "data": result['pooled_average']})
                    all_subject_sem.append({"type": "average", "data": result['pooled_sem']})
                
                total_sessions += len(result['session_dates'])
        
        # Check if we have data to plot
        if not all_subject_averages:
            print("No valid data found for cross-subject averaging")
            return None
        
        # Create figure for cross-subject analysis
        plt.figure(figsize=(12, 7))
        
        if win_loss:
            # Separate rewarded and unrewarded data
            rewarded_averages = [item["data"] for item in all_subject_averages if item["type"] == "rewarded"]
            unrewarded_averages = [item["data"] for item in all_subject_averages if item["type"] == "unrewarded"]
            
            if rewarded_averages:
                # Calculate mean and SEM across subjects for rewarded trials
                rewarded_mean = np.mean(rewarded_averages, axis=0)
                rewarded_sem = np.std(rewarded_averages, axis=0) / np.sqrt(len(rewarded_averages))
                
                plt.fill_between(time_axis, 
                               rewarded_mean - rewarded_sem,  
                               rewarded_mean + rewarded_sem,  
                               color='lightgreen', alpha=0.4, 
                               label=f'_Rewarded ± SEM (n={len(rewarded_averages)} subjects)')  
                plt.plot(time_axis, rewarded_mean, color='green', linewidth=2.5, label=f'Rew Avg')
            
            if unrewarded_averages:
                # Calculate mean and SEM across subjects for unrewarded trials
                unrewarded_mean = np.mean(unrewarded_averages, axis=0)
                unrewarded_sem = np.std(unrewarded_averages, axis=0) / np.sqrt(len(unrewarded_averages))
                
                plt.fill_between(time_axis, 
                               unrewarded_mean - unrewarded_sem,  
                               unrewarded_mean + unrewarded_sem,  
                               color='lightsalmon', alpha=0.4, 
                               label=f'_Unrewarded ± SEM (n={len(unrewarded_averages)} subjects)')  
                plt.plot(time_axis, unrewarded_mean, color='darkorange', linewidth=2.5, label=f'Unrew Avg')
        else:
            # Get all average data
            all_averages = [item["data"] for item in all_subject_averages if item["type"] == "average"]
            
            # Calculate mean and SEM across subjects
            across_subject_mean = np.mean(all_averages, axis=0)
            across_subject_sem = np.std(all_averages, axis=0) / np.sqrt(len(all_averages))
            
            plt.fill_between(time_axis,
                           across_subject_mean - across_subject_sem,  
                           across_subject_mean + across_subject_sem,  
                           color='lightgreen', alpha=0.4,
                           label=f'Mean ± SEM (n={len(specific_subjects)} subjects)')  
            plt.plot(time_axis, across_subject_mean, color='green', linewidth=2.5, label='Overall Avg')
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Labels and formatting
        plt.xlabel('Time (s) after first lick', fontsize=24)
        plt.ylabel('z-ΔF/F', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.title(f'Pooled Photometry Response: All Subjects (n={len(specific_subjects)})', fontsize=26)
        plt.xlim([-pre_cue_time, post_cue_time])
        plt.legend(loc='upper right', fontsize=24)
        plt.tight_layout()
        
        # Save the figure
        save_figure(plt.gcf(), "all_subjects", "pooled", f"pooled_results{'_winloss' if win_loss else ''}")
        
        plt.show()

    else:
        # Original single-subject behavior
        print(f"Analyzing pooled results for subject {subject_id}...")
        return analyze_pooled_data(subject_id, win_loss=win_loss, force_recompute=force_recompute,
                               show_session_traces=show_session_traces, behavior_df=behavior_df)


def all_results(win_loss=False, force_recompute=False):
    """Analyze and visualize results for all subjects"""
    print("Analyzing all subjects...")
    return analyze_all_subjects(win_loss=win_loss, force_recompute=force_recompute)

def analyze_group_reward_rates(behavior_df=None, min_trials=100, save_fig=True, subjids=None):
    """
    Calculate and plot average reward rates across sessions for all subjects chronologically.

    Parameters:
    -----------
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavioral dataframe to analyze. If None, will load from parquet.
    min_trials : int, optional
        Minimum number of trials required to include a session (default=100)
    save_fig : bool, optional
        Whether to save the generated figure (default=True)
    subjids : list, optional
        List of subject IDs to include in the analysis. If None, use all subjects.

    Returns:
    --------
    dict: Dictionary with reward rate data by subject and session
    """
    # Load behavior data if not provided
    if behavior_df is None:
        try:
            behavior_df = load_filtered_behavior_data("MatchingPennies")
            print(f"Loaded behavior data from parquet file: {len(behavior_df)} rows")
        except Exception as e:
            print(f"Error loading behavior data: {e}")
            return None

    # Prepare storage for results
    results = {
        'subjects': [],
        'session_reward_rates': {},
        'max_sessions': 0,
        'subject_avg_rates': {},  # Store average reward rate per subject
        'subject_session_counts': {}  # Store session count per subject
    }

    # Get unique subjects, filtered by provided list if specified
    all_subjects = behavior_df['subjid'].unique()
    if subjids is not None:
        subjects = [s for s in all_subjects if s in subjids]
        print(f"Filtering {len(all_subjects)} subjects to {len(subjects)} specified subjects")
    else:
        subjects = all_subjects

    results['subjects'] = list(subjects)
    print(f"Analyzing {len(subjects)} subjects")

    # Process each subject
    for subject_id in subjects:
        print(f"Processing {subject_id}...")
        subject_data = behavior_df[behavior_df['subjid'] == subject_id]

        # Sort sessions chronologically
        sessions = sorted(subject_data['date'].unique())
        session_reward_rates = []

        for session_date in sessions:
            session_df = subject_data[subject_data['date'] == session_date]

            # Skip sessions with too few trials
            if len(session_df) < min_trials:
                print(f"  Skipping {subject_id}/{session_date}, fewer than {min_trials} trials ({len(session_df)})")
                continue

            # Get rewards for this session
            rewards = session_df['reward'].values
            window_size = 20
            reward_rates = []
            overall_rate = np.mean(rewards)

            # Calculate moving average reward rate
            for i in range(len(rewards)):
                if i < window_size:
                    available_data = rewards[:i + 1]
                    missing_data_weight = (window_size - len(available_data)) / window_size
                    rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
                else:
                    rate = np.mean(rewards[i - window_size + 1:i + 1])
                reward_rates.append(rate)

            # Calculate average reward rate for the session
            session_avg_rate = np.mean(reward_rates)
            session_reward_rates.append(session_avg_rate)
            print(
                f"  {subject_id}/{session_date}: Average reward rate = {session_avg_rate:.3f} ({len(session_df)} trials)")

        # Store session data for this subject
        results['session_reward_rates'][subject_id] = session_reward_rates
        results['max_sessions'] = max(results['max_sessions'], len(session_reward_rates))

        # Calculate and store the average reward rate across all sessions for this subject
        if session_reward_rates:
            avg_reward_rate = np.mean(session_reward_rates)
            results['subject_avg_rates'][subject_id] = avg_reward_rate
            results['subject_session_counts'][subject_id] = len(session_reward_rates)

    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Initialize data structure for the group average
    max_sessions = results['max_sessions']
    sessions_data = [[] for _ in range(max_sessions + 1)]  # +1 to handle 1-indexed session numbers
    
    # Plot each subject's reward rate progression in thin gray
    for subject_id in subjects:
        rates = results['session_reward_rates'][subject_id]
        if rates:
            # Plot individual subject in thin gray line
            plt.plot(range(1, len(rates) + 1), rates, '-', 
                     color='gray', linewidth=0.8, alpha=0.5)
            
            # Store data for group average calculation
            for i, rate in enumerate(rates):
                sessions_data[i+1].append(rate)

    # Calculate and plot the group average with SEM
    x_values = []
    mean_values = []
    sem_values = []
    
    for session_num, session_rates in enumerate(sessions_data):
        if session_num == 0:
            continue  # Skip the dummy index 0
            
        # Only include session in group average if at least 3 subjects have data
        if len(session_rates) >= 3:
            x_values.append(session_num)
            session_mean = np.mean(session_rates)
            session_sem = np.std(session_rates) / np.sqrt(len(session_rates))
            mean_values.append(session_mean)
            sem_values.append(session_sem)

    # Plot the group average line and SEM if we have data
    if x_values:
        plt.fill_between(x_values, 
                         np.array(mean_values) - np.array(sem_values),
                         np.array(mean_values) + np.array(sem_values),
                         color='black', alpha=0.2)
        plt.plot(x_values, mean_values, 
                 color='black', linewidth=2.5, label=f'Group average (n={len(subjects)})')

    # Add a reference line at 0.5
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5)

    # Set axis labels and title
    plt.xlabel('Session Number', fontsize=26)
    plt.ylabel('Average Reward Rate', fontsize=26)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title('Reward Rate Across Learning', fontsize=26)

    # Set axis limits
    plt.xlim(0.5, results['max_sessions'] + 0.5)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add legend for group average
    plt.legend(loc='upper right', fontsize=24)

    plt.tight_layout()

    # Save the figure
    if save_fig:
        figure_path = os.path.join(output_dir, "group_analysis")
        ensure_directory_exists(figure_path)
        fig_file = os.path.join(figure_path, f"group_reward_rates.png")

        try:
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_file}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    plt.show()
    
    # Add group average to results
    results['group_average'] = {
        'x': x_values,
        'mean': mean_values,
        'sem': sem_values
    }
    
    return results


def analyze_group_computer_confidence(behavior_df=None, min_trials=100, save_fig=True, subjids=None):
    """
    Calculate and plot average computer confidence across sessions for all subjects chronologically.

    Parameters:
    -----------
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavioral dataframe to analyze. If None, will load from parquet.
    min_trials : int, optional
        Minimum number of trials required to include a session (default=100)
    save_fig : bool, optional
        Whether to save the generated figure (default=True)
    subjids : list, optional
        List of subject IDs to include in the analysis. If None, use all subjects.

    Returns:
    --------
    dict: Dictionary with computer confidence data by subject and session
    """
    # Load behavior data if not provided
    if behavior_df is None:
        try:
            behavior_df = load_filtered_behavior_data("MatchingPennies")
            print(f"Loaded behavior data from parquet file: {len(behavior_df)} rows")
        except Exception as e:
            print(f"Error loading behavior data: {e}")
            return None

    # Check if min_pvalue is in the dataframe
    if 'min_pvalue' not in behavior_df.columns:
        print("Error: 'min_pvalue' column not found in behavior data")
        return None

    # Prepare storage for results
    results = {
        'subjects': [],
        'session_confidence': {},
        'max_sessions': 0,
        'subject_avg_confidence': {},  # Store average confidence per subject
        'subject_session_counts': {}  # Store session count per subject
    }

    # Get unique subjects, filtered by provided list if specified
    all_subjects = behavior_df['subjid'].unique()
    if subjids is not None:
        subjects = [s for s in all_subjects if s in subjids]
        print(f"Filtering {len(all_subjects)} subjects to {len(subjects)} specified subjects")
    else:
        subjects = all_subjects

    results['subjects'] = list(subjects)
    print(f"Analyzing {len(subjects)} subjects")

    # Process each subject
    for subject_id in subjects:
        print(f"Processing {subject_id}...")
        subject_data = behavior_df[behavior_df['subjid'] == subject_id]

        # Sort sessions chronologically
        sessions = sorted(subject_data['date'].unique())
        session_confidence = []

        for session_date in sessions:
            session_df = subject_data[subject_data['date'] == session_date]

            # Skip sessions with too few trials
            if len(session_df) < min_trials:
                print(f"  Skipping {subject_id}/{session_date}, fewer than {min_trials} trials ({len(session_df)})")
                continue

            # Get p-values for this session
            p_values = session_df['min_pvalue'].values

            # Replace NaN values with 1.0 (no confidence)
            p_values = np.nan_to_num(p_values, nan=1.0)

            # Apply cutoff for very small p-values (p < 10^-12)
            min_p_value = 1e-12
            p_values = np.maximum(p_values, min_p_value)

            # Calculate confidence as -log10(p)
            confidence = -np.log10(p_values)

            # Calculate moving average confidence with window size 20
            window_size = 20
            confidence_values = []
            overall_confidence = np.mean(confidence)

            # Calculate moving average
            for i in range(len(confidence)):
                if i < window_size:
                    available_data = confidence[:i + 1]
                    missing_data_weight = (window_size - len(available_data)) / window_size
                    conf = (np.sum(
                        available_data) + missing_data_weight * window_size * overall_confidence) / window_size
                else:
                    conf = np.mean(confidence[i - window_size + 1:i + 1])
                confidence_values.append(conf)

            # Calculate average confidence for the session
            session_avg_conf = np.mean(confidence_values)
            session_confidence.append(session_avg_conf)
            print(
                f"  {subject_id}/{session_date}: Average confidence = {session_avg_conf:.3f} ({len(session_df)} trials)")

        # Store session data for this subject
        results['session_confidence'][subject_id] = session_confidence
        results['max_sessions'] = max(results['max_sessions'], len(session_confidence))

        # Calculate and store the average confidence across all sessions for this subject
        if session_confidence:
            avg_confidence = np.mean(session_confidence)
            results['subject_avg_confidence'][subject_id] = avg_confidence
            results['subject_session_counts'][subject_id] = len(session_confidence)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Initialize data structure for the group average
    max_sessions = results['max_sessions']
    sessions_data = [[] for _ in range(max_sessions + 1)]  # +1 to handle 1-indexed session numbers
    
    # Plot each subject's confidence progression in thin gray
    for subject_id in subjects:
        confidence = results['session_confidence'][subject_id]
        if confidence:
            # Plot individual subject in thin gray line
            plt.plot(range(1, len(confidence) + 1), confidence, '-', 
                     color='gray', linewidth=0.8, alpha=0.5)
            
            # Store data for group average calculation
            for i, conf in enumerate(confidence):
                sessions_data[i+1].append(conf)

    # Calculate and plot the group average with SEM
    x_values = []
    mean_values = []
    sem_values = []
    
    for session_num, session_confs in enumerate(sessions_data):
        if session_num == 0:
            continue  # Skip the dummy index 0
            
        # Only include session in group average if at least 3 subjects have data
        if len(session_confs) >= 3:
            x_values.append(session_num)
            session_mean = np.mean(session_confs)
            session_sem = np.std(session_confs) / np.sqrt(len(session_confs))
            mean_values.append(session_mean)
            sem_values.append(session_sem)

    # Plot the group average line and SEM if we have data
    if x_values:
        plt.fill_between(x_values, 
                         np.array(mean_values) - np.array(sem_values),
                         np.array(mean_values) + np.array(sem_values),
                         color='black', alpha=0.2)
        plt.plot(x_values, mean_values,
                 color='black', linewidth=2.5, label=f'Group average (n={len(subjects)})')

    # Add a reference line at p=0.05 (-log10(0.05) ≈ 1.3)
    p_05_line = -np.log10(0.05)
    plt.axhline(y=p_05_line, color='red', linestyle='--', linewidth=1.5)

    # Set axis labels and title
    plt.xlabel('Session Number', fontsize=24)
    plt.ylabel('Computer "Confidence" (-log10(min_p))', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title('Average Computer Confidence Across Learning', fontsize=26)

    # Set axis limits
    plt.xlim(0.5, results['max_sessions'] + 0.5)
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.legend(loc='upper right', fontsize=24)

    plt.tight_layout()

    # Save the figure
    if save_fig:
        figure_path = os.path.join(output_dir, "group_analysis")
        ensure_directory_exists(figure_path)
        fig_file = os.path.join(figure_path, f"group_computer_confidence.png")

        try:
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_file}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    plt.show()
    
    # Add group average to results
    results['group_average'] = {
        'x': x_values,
        'mean': mean_values,
        'sem': sem_values
    }
    
    return results

def plot_single_session_metrics(subject_id, session_date, metric_type="reward_rate", behavior_df=None, window_size=20):
    """
    Plot reward rate or computer confidence metrics for a single session in publication-ready format.
    
    Parameters:
    -----------
    subject_id : str
        The subject ID to analyze
    session_date : str
        The specific session date to plot (format: 'YYYYMMDD')
    metric_type : str
        Either "reward_rate" or "computer_confidence"
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use
    """
    if behavior_df is None:
        behavior_df = load_filtered_behavior_data("MatchingPennies")
    
    # Filter data for the specific subject and session
    session_data = behavior_df[(behavior_df['subjid'] == subject_id) & 
                              (behavior_df['date'] == session_date)]
    
    if session_data.empty:
        print(f"No data found for {subject_id} on {session_date}")
        return
    
    # Calculate the selected metric
    window_size = window_size
    metric_values = []
    
    if metric_type == "reward_rate":
        # Calculate reward rates with moving average
        rewards = session_data['reward'].values
        overall_rate = np.mean(rewards)
        
        for i in range(len(rewards)):
            if i < window_size:
                available_data = rewards[:i + 1]
                missing_data_weight = (window_size - len(available_data)) / window_size
                rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
            else:
                rate = np.mean(rewards[i - window_size + 1:i + 1])
            metric_values.append(rate)
        
        y_label = "Reward Rate"
        threshold = 0.5
        threshold_label = None
        threshold_style = {'color': 'black', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.7}
        
    else:  # computer_confidence
        # Calculate computer confidence (-log10(p))
        p_values = session_data['min_pvalue'].values
        p_values = np.nan_to_num(p_values, nan=1.0)
        p_values = np.maximum(p_values, 1e-12)  # Apply minimum p-value threshold
        confidence = -np.log10(p_values)
        overall_confidence = np.mean(confidence)
        
        for i in range(len(confidence)):
            if i < window_size:
                available_data = confidence[:i + 1]
                missing_data_weight = (window_size - len(available_data)) / window_size
                conf = (np.sum(available_data) + missing_data_weight * window_size * overall_confidence) / window_size
            else:
                conf = np.mean(confidence[i - window_size + 1:i + 1])
            metric_values.append(conf)
        
        y_label = "Computer 'Confidence'\n(-log10(min_p))"
        threshold = -np.log10(0.05)  # p = 0.05
        threshold_style = {'color': 'red', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7}

    # Create figure with publication-quality styling
    plt.figure(figsize=(12, 4))
    
    # Plot the data with a clean black line
    trials = range(1, len(metric_values) + 1)
    plt.plot(trials, metric_values, color='black', linewidth=2)
    
    # Add threshold line with appropriate styling
    plt.axhline(y=threshold, **threshold_style)
    
    # Set clean, professional axis labels and title
    plt.xlabel('Trial Number', fontsize=24)
    plt.ylabel(y_label, fontsize=21)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.title(f'{subject_id}: Session {session_date}', fontsize=26, fontweight='bold')
    
    # Configure grid and spines for publication style
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    
    # Set reasonable y-axis limits
    if metric_type == "reward_rate":
        plt.ylim([0, 1])
    else:
        y_max = max(12, max(metric_values) * 1.1)  # Cap at reasonable maximum
        plt.ylim([0, y_max])
    
    # Add legend if threshold has a label
    if threshold_label:
        plt.legend(frameon=True, fontsize=12)
    
    plt.tight_layout()
    
    # Save figure with high resolution
    fig_name = f"{subject_id}_{session_date}_{metric_type}"
    plt.savefig(f"{fig_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{fig_name}.pdf", format='pdf', bbox_inches='tight')
    
    plt.show()

def analyze_reward_rate_quartiles(subject_id, session_date=None, win_loss=False, behavior_df=None, specific_subjects=None):
    """
    Analyze photometry signals binned by reward rate quartiles for a single subject or across subjects
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject, or "All" for cross-subject analysis
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    win_loss : bool, optional
        Whether to split by rewarded/unrewarded trials
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    specific_subjects : list, optional
        List of subject IDs to include if subject_id="All"
        
    Returns:
    --------
    dict: Analysis results including quartile bins and signal data
    """
    # Handle cross-subject analysis
    if subject_id == "All":
        if specific_subjects is None:
            # Default list of subjects
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Store data for each quartile across subjects
        quartile_data = {
            0: {'rewarded': [], 'unrewarded': [], 'all': []},
            1: {'rewarded': [], 'unrewarded': [], 'all': []},
            2: {'rewarded': [], 'unrewarded': [], 'all': []},
            3: {'rewarded': [], 'unrewarded': [], 'all': []}
        }
        
        # Track quartile averages by subject for overall statistics
        subject_quartile_avgs = []
        time_axis = None
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for reward rate quartile analysis...")
            
            # Call the single subject analysis function for each subject
            # We'll collect the individual results for later averaging
            subj_result = analyze_reward_rate_quartiles_single(subj, session_date, win_loss, behavior_df)
            
            if subj_result and 'quartile_bins' in subj_result and 'reward_rates' in subj_result:
                # Extract the quartile bins and reward rates from this subject
                quartile_bins = subj_result['quartile_bins']
                reward_rates = subj_result['reward_rates']
                quartile_avgs = subj_result['quartile_averages']
                
                # Store quartile averages for this subject
                subject_quartile_avgs.append(quartile_avgs)
                
                # Get subject-specific quartile data
                # We need to re-process the subject to get the photometry signals by quartile
                # (since the original function doesn't return these directly)
                subject_path = os.path.join(base_dir, subj)
                
                # Re-run the necessary parts of the analysis to get the photometry signals
                # This is similar to logic in analyze_reward_rate_quartiles_single
                all_plotting_data = []
                all_reward_rates = []
                all_reward_outcomes = []
                
                # Get all sessions for this subject
                matching_pennies_sessions = set()
                try:
                    if behavior_df is not None:
                        subject_data = behavior_df[behavior_df['subjid'] == subj]
                        matching_pennies_sessions = set(subject_data['date'].unique())
                    else:
                        df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                        df['date'] = df['date'].astype(str)
                        subject_data = df[(df['subjid'] == subj) & (df['protocol'].str.contains('MatchingPennies', na=False))]
                        matching_pennies_sessions = set(subject_data['date'].unique())
                except Exception as e:
                    print(f"Warning: Could not load session info for {subj}: {e}")
                    continue

                sessions = sorted([d for d in os.listdir(subject_path)
                        if os.path.isdir(os.path.join(subject_path, d)) and
                        os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                        d in matching_pennies_sessions])
                
                # Process each session separately to get photometry data
                for sess_date in sessions:
                    session_result = process_session(subj, sess_date, behavior_df=behavior_df)
                    if not session_result:
                        continue

                    if len(session_result['non_m_trials']) < 100:
                        print(f"Skipping {subj}/{sess_date}, less than 100 valid trials.")
                        continue

                    # Calculate reward rates
                    behavior_data = session_result['behavioral_data']
                    rewards = np.array(behavior_data['reward'])
                    window_size = 20
                    reward_rates = []
                    overall_rate = np.mean(rewards)

                    for i in range(len(rewards)):
                        if i < window_size:
                            available_data = rewards[:i + 1]
                            missing_data_weight = (window_size - len(available_data)) / window_size
                            rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
                        else:
                            rate = np.mean(rewards[i - window_size + 1:i + 1])
                        reward_rates.append(rate)

                    # Get valid trials
                    non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                            if idx in session_result["non_m_trials"]])
                    
                    # Store data and corresponding reward rates
                    all_plotting_data.append(session_result['plotting_data'])
                    all_reward_rates.extend(np.array(reward_rates)[non_m_indices])
                    all_reward_outcomes.append(session_result["reward_outcomes"][non_m_indices])
                    
                    if time_axis is None:
                        time_axis = session_result['time_axis']
                
                # Skip if no valid sessions for this subject
                if not all_plotting_data:
                    print(f"No valid sessions found for {subj}")
                    continue
                    
                # Process the data
                plotting_data = np.vstack(all_plotting_data)
                reward_rates = np.array(all_reward_rates)
                reward_outcomes = np.concatenate(all_reward_outcomes)
                
                # Create quartile bins for this subject
                quartile_bins = pd.qcut(reward_rates, q=4, labels=False)
                
                # Calculate and store average signals for each quartile for this subject
                for quartile in range(4):
                    quartile_trials = quartile_bins == quartile
                    
                    # Split by win/loss if requested
                    if win_loss:
                        # Rewarded trials in this quartile
                        quartile_rewarded = (quartile_bins == quartile) & (reward_outcomes == 1)
                        if np.sum(quartile_rewarded) > 0:
                            rewarded_avg = np.mean(plotting_data[quartile_rewarded], axis=0)
                            quartile_data[quartile]['rewarded'].append(rewarded_avg)
                        
                        # Unrewarded trials in this quartile
                        quartile_unrewarded = (quartile_bins == quartile) & (reward_outcomes == 0)
                        if np.sum(quartile_unrewarded) > 0:
                            unrewarded_avg = np.mean(plotting_data[quartile_unrewarded], axis=0)
                            quartile_data[quartile]['unrewarded'].append(unrewarded_avg)
                    else:
                        # All trials in this quartile
                        if np.sum(quartile_trials) > 0:
                            quartile_avg = np.mean(plotting_data[quartile_trials], axis=0)
                            quartile_data[quartile]['all'].append(quartile_avg)
        
        # Check if we have data to plot
        if time_axis is None:
            print("No valid data found for analysis")
            return None
        
        # Calculate average quartile reward rates across subjects
        if subject_quartile_avgs:
            quartile_avgs = np.mean(subject_quartile_avgs, axis=0)
            quartile_sems = np.std(subject_quartile_avgs, axis=0) / np.sqrt(len(subject_quartile_avgs))
        
        # Create the plot
        plt.figure(figsize=(12, 7))
        colors = ['blue', 'green', 'orange', 'red']
        
        # Plot data based on win/loss parameter
        if win_loss:
            for quartile in range(4):
                # Plot rewarded trials
                if quartile_data[quartile]['rewarded']:
                    rewarded_avgs = np.array(quartile_data[quartile]['rewarded'])
                    rewarded_mean = np.mean(rewarded_avgs, axis=0)
                    rewarded_sem = np.std(rewarded_avgs, axis=0) / np.sqrt(len(rewarded_avgs))
                    
                    plt.fill_between(time_axis, rewarded_mean - rewarded_sem,
                                   rewarded_mean + rewarded_sem, 
                                   color=colors[quartile], alpha=0.15)
                    plt.plot(time_axis, rewarded_mean,
                           color=colors[quartile], linewidth=2,
                           label=f'Q{quartile+1} Rewarded (n={len(rewarded_avgs)} subjects)')
                
                # Plot unrewarded trials
                if quartile_data[quartile]['unrewarded']:
                    unrewarded_avgs = np.array(quartile_data[quartile]['unrewarded'])
                    unrewarded_mean = np.mean(unrewarded_avgs, axis=0)
                    unrewarded_sem = np.std(unrewarded_avgs, axis=0) / np.sqrt(len(unrewarded_avgs))
                    
                    plt.plot(time_axis, unrewarded_mean,
                           color=colors[quartile], linewidth=2, linestyle='--',
                           label=f'Q{quartile+1} Unrewarded (n={len(unrewarded_avgs)} subjects)')
        else:
            # Plot all trials
            for quartile in range(4):
                if quartile_data[quartile]['all']:
                    all_avgs = np.array(quartile_data[quartile]['all'])
                    all_mean = np.mean(all_avgs, axis=0)
                    all_sem = np.std(all_avgs, axis=0) / np.sqrt(len(all_avgs))
                    
                    plt.fill_between(time_axis, all_mean - all_sem,
                                   all_mean + all_sem, 
                                   color=colors[quartile], alpha=0.15)
                    plt.plot(time_axis, all_mean,
                           color=colors[quartile], linewidth=2,
                           label=f'Q{quartile+1} (n={len(all_avgs)} subjects)')
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Create custom legend with integrated Low-High labels
        legend_handles = []
        if win_loss:
            # First add solid/dashed line explanation
            solid_line = plt.Line2D([0], [0], color='black', linewidth=2, label='Rewarded')
            dashed_line = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Unrewarded')
            legend_handles.extend([solid_line, dashed_line])
            
            # Add spacer
            legend_handles.append(plt.Line2D([0], [0], color='none', label=''))
        
        # Add color-coded quartiles
        legend_handles.append(plt.Line2D([0], [0], color=colors[0], linewidth=2, label='Q1   Low Reward Rate'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[1], linewidth=2, label='Q2   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[2], linewidth=2, label='Q3   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[3], linewidth=2, label='Q4   High Reward Rate'))
        
        # Add legend
        plt.legend(handles=legend_handles, loc='upper right', fontsize=12, 
                 title="Reward Rate", title_fontsize=12)
        
        # Add labels and title
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('ΔF/F', fontsize=16)
        plt.title(f'Pooled Photometry by Reward Rate Quartiles: All Subjects (n={len(specific_subjects)})', 
                 fontsize=20)
        plt.xlim([-pre_cue_time, post_cue_time])
        
        # Add text with quartile averages at the bottom of the plot
        quartile_text = "Average reward rates: " + ", ".join([f"Q{q+1}: {quartile_avgs[q]:.3f}±{quartile_sems[q]:.3f}" for q in range(4)])
        plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text
        
        # Save the figure
        save_figure(plt.gcf(), "all_subjects", "pooled", 
                  f"reward_rate_quartiles{'_winloss' if win_loss else ''}")
        
        plt.show()
        
        return {
            'subject_id': 'All',
            'specific_subjects': specific_subjects,
            'quartile_data': quartile_data,
            'quartile_avgs': quartile_avgs,
            'quartile_sems': quartile_sems,
            'time_axis': time_axis,
            'win_loss': win_loss
        }
    else:
        # Original single-subject behavior
        return analyze_reward_rate_quartiles_single(subject_id, session_date, win_loss, behavior_df)


def analyze_reward_rate_quartiles_single(subject_id, session_date=None, win_loss=False, behavior_df=None):
    """
    Analyze photometry signals binned by reward rate quartiles for a single session or pooled across sessions
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    win_loss : bool, optional
        Whether to split by rewarded/unrewarded trials
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
        
    Returns:
    --------
    dict: Analysis results including quartile bins and signal data
    """
    original_session_date = session_date  
    all_plotting_data = []
    all_reward_rates = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
        matching_pennies_sessions = set()
        try:
            if behavior_df is not None:
                # If behavior_df is provided, filter it for this subject
                subject_data = behavior_df[behavior_df['subjid'] == subject_id]
                matching_pennies_sessions = set(subject_data['date'].unique())
                print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
            else:
                # Otherwise load from parquet file
                df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                df['date'] = df['date'].astype(str)
                subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
                matching_pennies_sessions = set(subject_data['date'].unique())
                print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
        except Exception as e:
            print(f"Warning: Could not load session info: {e}")

        sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])
        
        # Process each session separately to maintain session-specific reward rate context
        for session_date in sessions:
            # Pass behavior_df to process_session to reuse data
            session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Calculate reward rates for this session
            behavior_data = session_result['behavioral_data']
            rewards = np.array(behavior_data['reward'])
            window_size = 20
            reward_rates = []
            overall_rate = np.mean(rewards)

            for i in range(len(rewards)):
                if i < window_size:
                    available_data = rewards[:i + 1]
                    missing_data_weight = (window_size - len(available_data)) / window_size
                    rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
                else:
                    rate = np.mean(rewards[i - window_size + 1:i + 1])
                reward_rates.append(rate)

            # Get valid trials
            non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                    if idx in session_result["non_m_trials"]])
            
            # Store data and corresponding reward rates
            all_plotting_data.append(session_result['plotting_data'])
            all_reward_rates.extend(np.array(reward_rates)[non_m_indices])
            all_reward_outcomes.append(session_result["reward_outcomes"][non_m_indices])
            time_axis = session_result['time_axis']
        
        plot_title = f'Pooled Photometry by Reward Rate Quartiles: {subject_id}'
        plotting_data = np.vstack(all_plotting_data)
        reward_rates = np.array(all_reward_rates)
        reward_outcomes = np.concatenate(all_reward_outcomes)

    else:
        # Single session analysis
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        window_size = 20
        reward_rates = []
        overall_rate = np.mean(rewards)

        for i in range(len(rewards)):
            if i < window_size:
                available_data = rewards[:i + 1]
                missing_data_weight = (window_size - len(available_data)) / window_size
                rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
            else:
                rate = np.mean(rewards[i - window_size + 1:i + 1])
            reward_rates.append(rate)

        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                if idx in session_result["non_m_trials"]])
        
        plotting_data = session_result['plotting_data']
        reward_rates = np.array(reward_rates)[non_m_indices]
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        time_axis = session_result['time_axis']
        plot_title = f'LC Signal by Reward Rate Quartiles: {subject_id} - {session_date}'

    # Create quartile bins based on all reward rates
    quartile_bins = pd.qcut(reward_rates, q=4, labels=False)
    
    # Calculate average reward rate for each quartile
    quartile_averages = []
    for quartile in range(4):
        quartile_avg = np.mean(reward_rates[quartile_bins == quartile])
        quartile_averages.append(quartile_avg)
    
    # Print average reward rates for each quartile
    print(f"\nAverage reward rates by quartile:")
    for quartile in range(4):
        print(f"Quartile {quartile + 1}: {quartile_averages[quartile]:.4f}")

    # Create the plot
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'red']
    
    # Track trial counts
    trial_counts = {
        'Q1': {'rewarded': 0, 'unrewarded': 0},
        'Q2': {'rewarded': 0, 'unrewarded': 0},
        'Q3': {'rewarded': 0, 'unrewarded': 0},
        'Q4': {'rewarded': 0, 'unrewarded': 0}
    }

    if win_loss:
        for quartile in range(4):
            quartile_rewarded = (quartile_bins == quartile) & (reward_outcomes == 1)
            quartile_unrewarded = (quartile_bins == quartile) & (reward_outcomes == 0)
            
            # Update trial counts
            trial_counts[f'Q{quartile + 1}']['rewarded'] = np.sum(quartile_rewarded)
            trial_counts[f'Q{quartile + 1}']['unrewarded'] = np.sum(quartile_unrewarded)

            if np.sum(quartile_rewarded) > 0:
                rewarded_avg = np.mean(plotting_data[quartile_rewarded], axis=0)
                rewarded_sem = calculate_sem(plotting_data[quartile_rewarded], axis=0)
                plt.fill_between(time_axis,
                               rewarded_avg - rewarded_sem,
                               rewarded_avg + rewarded_sem,
                               color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, rewarded_avg,
                        color=colors[quartile], linewidth=2,
                        label=f'Q{quartile + 1}')

            if np.sum(quartile_unrewarded) > 0:
                unrewarded_avg = np.mean(plotting_data[quartile_unrewarded], axis=0)
                unrewarded_sem = calculate_sem(plotting_data[quartile_unrewarded], axis=0)
                plt.plot(time_axis, unrewarded_avg,
                        color=colors[quartile], linewidth=2, linestyle='--',
                        label=f'_Q{quartile + 1}')
    else:
        for quartile in range(4):
            quartile_trials = quartile_bins == quartile
            trial_count = np.sum(quartile_trials)
            
            # Calculate rewarded and unrewarded counts for this quartile
            rewarded_count = np.sum(quartile_trials & (reward_outcomes == 1))
            unrewarded_count = np.sum(quartile_trials & (reward_outcomes == 0))
            trial_counts[f'Q{quartile + 1}']['rewarded'] = rewarded_count
            trial_counts[f'Q{quartile + 1}']['unrewarded'] = unrewarded_count
            
            if trial_count > 0:
                quartile_avg = np.mean(plotting_data[quartile_trials], axis=0)
                quartile_sem = calculate_sem(plotting_data[quartile_trials], axis=0)

                plt.fill_between(time_axis,
                               quartile_avg - quartile_sem,
                               quartile_avg + quartile_sem,
                               color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, quartile_avg,
                        color=colors[quartile], linewidth=2,
                        label=f'Q{quartile + 1}')

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('ΔF/F', fontsize=16)
    plt.title(plot_title, fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])
    
    # Custom legend with integrated Low-High labels and arrow
    if win_loss:
        # Create a list of custom legend handles
        legend_handles = []
        
        # First add solid/dashed line explanation
        solid_line = plt.Line2D([0], [0], color='black', linewidth=2, label='Rewarded')
        dashed_line = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Unrewarded')
        legend_handles.extend([solid_line, dashed_line])
        
        # Add spacer
        legend_handles.append(plt.Line2D([0], [0], color='none', label=''))
        
        # Create colored lines for quartiles with Low/High labels
        legend_handles.append(plt.Line2D([0], [0], color=colors[0], linewidth=2, label='Q1   Low Reward Rate'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[1], linewidth=2, label='Q2   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[2], linewidth=2, label='Q3   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[3], linewidth=2, label='Q4   High Reward Rate'))
        
        # Create the legend with all handles
        plt.legend(handles=legend_handles, loc='upper right', fontsize=12, 
                 title="Reward Rate", title_fontsize=12)
        
        # Add trial counts as text on the left side below the figure
        trial_count_text = "Trial counts:\n"
        for q in range(4):
            trial_count_text += f"Q{q+1}: {trial_counts[f'Q{q+1}']['rewarded']} rewarded, {trial_counts[f'Q{q+1}']['unrewarded']} unrewarded\n"
        plt.figtext(0.25, 0.02, trial_count_text, ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    else:
        # Custom legend for non-win/loss plots
        legend_handles = []
        
        # Create colored lines for quartiles with Low/High labels
        legend_handles.append(plt.Line2D([0], [0], color=colors[0], linewidth=2, label='Q1   Low Reward Rate'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[1], linewidth=2, label='Q2   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[2], linewidth=2, label='Q3   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[3], linewidth=2, label='Q4   High Reward Rate'))
        
        # Create the legend
        plt.legend(handles=legend_handles, loc='upper right', fontsize=12, 
                 title="Reward Rate", title_fontsize=12)
        
        # Add trial counts as text on the left side below the figure
        trial_count_text = "Trial counts:\n"
        for q in range(4):
            trial_count_text += f"Q{q+1}: {trial_counts[f'Q{q+1}']['rewarded']} rewarded, {trial_counts[f'Q{q+1}']['unrewarded']} unrewarded\n"
        plt.figtext(0.25, 0.02, trial_count_text, ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add text with quartile averages at the right side bottom of the plot
    quartile_text = "Average reward rates:\n" + "\n".join([f"Q{q+1}: {avg:.3f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.75, 0.02, quartile_text, ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Make room for the text at the bottom

    # Save the figure
    fig_name = "reward_rate_quartiles"
    if original_session_date is None:
        save_figure(plt.gcf(), subject_id, "pooled", 
                f"{fig_name}{'_pooled'}{'_winloss' if win_loss else ''}")
    else:
        save_figure(plt.gcf(), subject_id, original_session_date, 
                f"{fig_name}{'_winloss' if win_loss else ''}")
        
    plt.show()
    
    return {
        'quartile_bins': quartile_bins,
        'reward_rates': reward_rates,
        'quartile_averages': quartile_averages
    }


def analyze_comp_confidence_quartiles(subject_id, session_date=None, win_loss=False, behavior_df=None, specific_subjects=None):
    """
    Analyze photometry signals binned by computer confidence quartiles for a single subject or across subjects
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject, or "All" for cross-subject analysis
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    win_loss : bool, optional
        Whether to split by rewarded/unrewarded trials
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    specific_subjects : list, optional
        List of subject IDs to include if subject_id="All"
        
    Returns:
    --------
    dict: Analysis results including quartile bins and confidence data
    """
    # Handle cross-subject analysis
    if subject_id == "All":
        if specific_subjects is None:
            # Default list of subjects
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Store data for each quartile across subjects
        quartile_data = {
            0: {'rewarded': [], 'unrewarded': [], 'all': []},
            1: {'rewarded': [], 'unrewarded': [], 'all': []},
            2: {'rewarded': [], 'unrewarded': [], 'all': []},
            3: {'rewarded': [], 'unrewarded': [], 'all': []}
        }
        
        # Track quartile averages by subject for overall statistics
        subject_quartile_avgs = []
        time_axis = None
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for computer confidence quartile analysis...")
            
            # Call the single subject analysis function for each subject
            # We'll collect the individual results for later averaging
            subj_result = analyze_comp_confidence_quartiles_single(subj, session_date, win_loss, behavior_df)
            
            if subj_result and 'quartile_bins' in subj_result and 'confidence_rates' in subj_result:
                # Extract the quartile bins and confidence values from this subject
                quartile_bins = subj_result['quartile_bins']
                confidence_rates = subj_result['confidence_rates']
                quartile_avgs = subj_result['quartile_averages']
                
                # Store quartile averages for this subject
                subject_quartile_avgs.append(quartile_avgs)
                
                # Get subject-specific quartile data
                # We need to re-process the subject to get the photometry signals by quartile
                # (since the original function doesn't return these directly)
                subject_path = os.path.join(base_dir, subj)
                
                # Re-run the necessary parts of the analysis to get the photometry signals
                # This is similar to logic in analyze_comp_confidence_quartiles_single
                all_plotting_data = []
                all_confidences = []
                all_reward_outcomes = []
                
                # Get all sessions for this subject
                matching_pennies_sessions = set()
                try:
                    if behavior_df is not None:
                        subject_data = behavior_df[behavior_df['subjid'] == subj]
                        matching_pennies_sessions = set(subject_data['date'].unique())
                    else:
                        df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                        df['date'] = df['date'].astype(str)
                        subject_data = df[(df['subjid'] == subj) & (df['protocol'].str.contains('MatchingPennies', na=False))]
                        matching_pennies_sessions = set(subject_data['date'].unique())
                except Exception as e:
                    print(f"Warning: Could not load session info for {subj}: {e}")
                    continue

                sessions = sorted([d for d in os.listdir(subject_path)
                        if os.path.isdir(os.path.join(subject_path, d)) and
                        os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                        d in matching_pennies_sessions])
                
                # Process each session separately to get photometry data
                for sess_date in sessions:
                    session_result = process_session(subj, sess_date, behavior_df=behavior_df)
                    if not session_result:
                        continue

                    if len(session_result['non_m_trials']) < 100:
                        print(f"Skipping {subj}/{sess_date}, less than 100 valid trials.")
                        continue

                    # Get p-value data for this session
                    try:
                        if behavior_df is not None:
                            # Simply filter from the already filtered dataframe
                            session_data = behavior_df[(behavior_df['subjid'] == subj) & 
                                                    (behavior_df['date'] == sess_date)]
                        else:
                            # Load from parquet file
                            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                            df['date'] = df['date'].astype(str)
                            session_data = df[(df['subjid'] == subj) & 
                                            (df['date'] == sess_date) & 
                                            (df["ignore"] == 0) & 
                                            (df['protocol'].str.contains('MatchingPennies', na=False))]

                        if session_data.empty:
                            print(f"No p-value data found for {subj} on {sess_date}")
                            continue
                            
                        # Extract p-values and calculate confidence
                        p_values = session_data['min_pvalue'].values
                        min_p_value = 1e-12
                        p_values = np.maximum(p_values, min_p_value)
                        confidence = -np.log10(p_values)

                        # Calculate moving average confidence
                        window_size = 20
                        confidence_rates = []
                        overall_confidence = np.mean(confidence)

                        for i in range(len(confidence)):
                            if i < window_size:
                                available_data = confidence[:i + 1]
                                missing_data_weight = (window_size - len(available_data)) / window_size
                                rate = (np.sum(available_data) + missing_data_weight * window_size * overall_confidence) / window_size
                            else:
                                rate = np.mean(confidence[i - window_size + 1:i + 1])
                            confidence_rates.append(rate)

                        # Get valid trials
                        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                                if idx in session_result["non_m_trials"]])
                        
                        # Store data and corresponding confidence rates
                        all_plotting_data.append(session_result['plotting_data'])
                        all_confidences.extend(np.array(confidence_rates)[non_m_indices])
                        all_reward_outcomes.append(session_result["reward_outcomes"][non_m_indices])
                        
                        if time_axis is None:
                            time_axis = session_result['time_axis']
                            
                    except Exception as e:
                        print(f"Error processing p-values for {subj}/{sess_date}: {e}")
                        continue
                
                # Skip if no valid sessions for this subject
                if not all_plotting_data:
                    print(f"No valid sessions found for {subj}")
                    continue
                    
                # Process the data
                plotting_data = np.vstack(all_plotting_data)
                confidence_rates = np.array(all_confidences)
                reward_outcomes = np.concatenate(all_reward_outcomes)
                
                # Create quartile bins for this subject
                quartile_bins = pd.qcut(confidence_rates, q=4, labels=False)
                
                # Calculate and store average signals for each quartile for this subject
                for quartile in range(4):
                    quartile_trials = quartile_bins == quartile
                    
                    # Split by win/loss if requested
                    if win_loss:
                        # Rewarded trials in this quartile
                        quartile_rewarded = (quartile_bins == quartile) & (reward_outcomes == 1)
                        if np.sum(quartile_rewarded) > 0:
                            rewarded_avg = np.mean(plotting_data[quartile_rewarded], axis=0)
                            quartile_data[quartile]['rewarded'].append(rewarded_avg)
                        
                        # Unrewarded trials in this quartile
                        quartile_unrewarded = (quartile_bins == quartile) & (reward_outcomes == 0)
                        if np.sum(quartile_unrewarded) > 0:
                            unrewarded_avg = np.mean(plotting_data[quartile_unrewarded], axis=0)
                            quartile_data[quartile]['unrewarded'].append(unrewarded_avg)
                    else:
                        # All trials in this quartile
                        if np.sum(quartile_trials) > 0:
                            quartile_avg = np.mean(plotting_data[quartile_trials], axis=0)
                            quartile_data[quartile]['all'].append(quartile_avg)
        
        # Check if we have data to plot
        if time_axis is None:
            print("No valid data found for analysis")
            return None
        
        # Calculate average quartile confidence values across subjects
        if subject_quartile_avgs:
            quartile_avgs = np.mean(subject_quartile_avgs, axis=0)
            quartile_sems = np.std(subject_quartile_avgs, axis=0) / np.sqrt(len(subject_quartile_avgs))
        
        # Create the plot
        plt.figure(figsize=(12, 7))
        # REVERSED color scheme compared to reward rate quartiles (red=Q1, blue=Q4)
        colors = ['red', 'orange', 'green', 'blue']
        
        # Plot data based on win/loss parameter
        if win_loss:
            for quartile in range(4):
                # Plot rewarded trials
                if quartile_data[quartile]['rewarded']:
                    rewarded_avgs = np.array(quartile_data[quartile]['rewarded'])
                    rewarded_mean = np.mean(rewarded_avgs, axis=0)
                    rewarded_sem = np.std(rewarded_avgs, axis=0) / np.sqrt(len(rewarded_avgs))
                    
                    plt.fill_between(time_axis, rewarded_mean - rewarded_sem,
                                   rewarded_mean + rewarded_sem, 
                                   color=colors[quartile], alpha=0.15)
                    plt.plot(time_axis, rewarded_mean,
                           color=colors[quartile], linewidth=2,
                           label=f'Q{quartile+1} Rewarded (n={len(rewarded_avgs)} subjects)')
                
                # Plot unrewarded trials
                if quartile_data[quartile]['unrewarded']:
                    unrewarded_avgs = np.array(quartile_data[quartile]['unrewarded'])
                    unrewarded_mean = np.mean(unrewarded_avgs, axis=0)
                    unrewarded_sem = np.std(unrewarded_avgs, axis=0) / np.sqrt(len(unrewarded_avgs))
                    
                    plt.plot(time_axis, unrewarded_mean,
                           color=colors[quartile], linewidth=2, linestyle='--',
                           label=f'Q{quartile+1} Unrewarded (n={len(unrewarded_avgs)} subjects)')
        else:
            # Plot all trials
            for quartile in range(4):
                if quartile_data[quartile]['all']:
                    all_avgs = np.array(quartile_data[quartile]['all'])
                    all_mean = np.mean(all_avgs, axis=0)
                    all_sem = np.std(all_avgs, axis=0) / np.sqrt(len(all_avgs))
                    
                    plt.fill_between(time_axis, all_mean - all_sem,
                                   all_mean + all_sem, 
                                   color=colors[quartile], alpha=0.15)
                    plt.plot(time_axis, all_mean,
                           color=colors[quartile], linewidth=2,
                           label=f'Q{quartile+1} (n={len(all_avgs)} subjects)')
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Create custom legend with integrated Low-High labels
        legend_handles = []
        if win_loss:
            # First add solid/dashed line explanation
            solid_line = plt.Line2D([0], [0], color='black', linewidth=2, label='Rewarded')
            dashed_line = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Unrewarded')
            legend_handles.extend([solid_line, dashed_line])
            
            # Add spacer
            legend_handles.append(plt.Line2D([0], [0], color='none', label=''))
        
        # Add color-coded quartiles - REVERSED order for computer confidence
        legend_handles.append(plt.Line2D([0], [0], color=colors[0], linewidth=2, label='Q1   Low Comp Conf'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[1], linewidth=2, label='Q2   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[2], linewidth=2, label='Q3   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[3], linewidth=2, label='Q4   High Comp Conf'))
        
        # Add legend
        plt.legend(handles=legend_handles, loc='upper right', fontsize=12, 
                 title="Computer Confidence", title_fontsize=12)
        
        # Add labels and title
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('ΔF/F', fontsize=16)
        plt.title(f'Pooled Photometry by Computer Confidence Quartiles: All Subjects (n={len(specific_subjects)})', 
                 fontsize=20)
        plt.xlim([-pre_cue_time, post_cue_time])
        
        # Add text with quartile averages at the bottom of the plot
        quartile_text = "Average confidence values: " + ", ".join([f"Q{q+1}: {quartile_avgs[q]:.4f}±{quartile_sems[q]:.4f}" for q in range(4)])
        plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text
        
        # Save the figure
        save_figure(plt.gcf(), "all_subjects", "pooled", 
                  f"computer_confidence_quartiles{'_winloss' if win_loss else ''}")
        
        plt.show()
        
        return {
            'subject_id': 'All',
            'specific_subjects': specific_subjects,
            'quartile_data': quartile_data,
            'quartile_avgs': quartile_avgs,
            'quartile_sems': quartile_sems,
            'time_axis': time_axis,
            'win_loss': win_loss
        }
    else:
        # Original single-subject behavior
        return analyze_comp_confidence_quartiles_single(subject_id, session_date, win_loss, behavior_df)

def analyze_comp_confidence_quartiles_single(subject_id, session_date=None, win_loss=False, behavior_df=None):
    """
    Analyze photometry signals binned by computer confidence quartiles for a single session or pooled across sessions

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    win_loss : bool, optional
        Whether to split by rewarded/unrewarded trials
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet

    Returns:
    --------
    dict: Analysis results including quartile bins and confidence values
    """
    original_session_date = session_date
    all_plotting_data = []
    all_confidences = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
        matching_pennies_sessions = set()
        try:
            if behavior_df is not None:
                # If behavior_df is provided, filter it for this subject
                subject_data = behavior_df[behavior_df['subjid'] == subject_id]
                matching_pennies_sessions = set(subject_data['date'].unique())
                print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
            else:
                # Otherwise load from parquet file
                df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                df['date'] = df['date'].astype(str)
                subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
                matching_pennies_sessions = set(subject_data['date'].unique())
                print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
        except Exception as e:
            print(f"Warning: Could not load session info: {e}")

        # Sort sessions chronologically, filtering to only include MatchingPennies sessions
        sessions = sorted([d for d in os.listdir(subject_path)
                    if os.path.isdir(os.path.join(subject_path, d)) and
                    os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                    d in matching_pennies_sessions])

        # Process each session separately to maintain session-specific confidence context
        for session_date in sessions:
            session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Get behavioral data for this session
            behavior_data = session_result['behavioral_data']

            # Get p-value data for this session
            try:
                if behavior_df is not None:
                    # Simply filter from the already filtered dataframe (no need to check protocol again)
                    session_data = behavior_df[(behavior_df['subjid'] == subject_id) & 
                                            (behavior_df['date'] == session_date)]
                else:
                    # Load from parquet file - this should rarely happen with your new approach
                    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                    df['date'] = df['date'].astype(str)
                    session_data = df[(df['subjid'] == subject_id) & 
                                    (df['date'] == session_date) & 
                                    (df["ignore"] == 0) & 
                                    (df['protocol'].str.contains('MatchingPennies', na=False))]

                if session_data.empty:
                    print(f"No p-value data found for {subject_id} on {session_date}")
                    continue
                # Extract p-values and calculate confidence
                p_values = session_data['min_pvalue'].values
                min_p_value = 1e-12
                p_values = np.maximum(p_values, min_p_value)
                confidence = -np.log10(p_values)

                # Calculate moving average confidence with window size 15
                window_size = 20
                confidence_rates = []
                overall_confidence = np.mean(confidence)

                for i in range(len(confidence)):
                    if i < window_size:
                        available_data = confidence[:i + 1]
                        missing_data_weight = (window_size - len(available_data)) / window_size
                        rate = (np.sum(
                            available_data) + missing_data_weight * window_size * overall_confidence) / window_size
                    else:
                        rate = np.mean(confidence[i - window_size + 1:i + 1])
                    confidence_rates.append(rate)

                # Get valid trials
                non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                          if idx in session_result["non_m_trials"]])

                # Store data and corresponding confidence rates
                all_plotting_data.append(session_result['plotting_data'])
                all_confidences.extend(np.array(confidence_rates)[non_m_indices])
                all_reward_outcomes.append(session_result["reward_outcomes"][non_m_indices])
                time_axis = session_result['time_axis']

            except Exception as e:
                print(f"Error processing p-values for {subject_id}/{session_date}: {e}")
                continue

        plot_title = f'Pooled Photometry by Computer Confidence Quartiles: {subject_id}'
        plotting_data = np.vstack(all_plotting_data)
        confidence_rates = np.array(all_confidences)
        reward_outcomes = np.concatenate(all_reward_outcomes)

    else:
        # Single session analysis
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        # Get p-value data for this session
        try:
            if behavior_df is not None:
                # Filter from provided dataframe
                session_data = behavior_df[(behavior_df['subjid'] == subject_id) & 
                                          (behavior_df['date'] == session_date) & 
                                          (behavior_df["ignore"] == 0) & 
                                          (behavior_df['protocol'].str.contains('MatchingPennies', na=False))]
            else:
                # Load from parquet file
                df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                df['date'] = df['date'].astype(str)  # Ensure date is a string
                session_data = df[(df['subjid'] == subject_id) & 
                                 (df['date'] == session_date) & 
                                 (df["ignore"] == 0) & 
                                 (df['protocol'].str.contains('MatchingPennies', na=False))]

            if session_data.empty:
                print(f"No p-value data found for {subject_id} on {session_date}")
                return None

            # Extract p-values and calculate confidence
            p_values = session_data['min_pvalue'].values
            min_p_value = 1e-12
            p_values = np.maximum(p_values, min_p_value)
            confidence = -np.log10(p_values)

            # Calculate moving average confidence with window size 15
            window_size = 20
            confidence_rates = []
            overall_confidence = np.mean(confidence)

            for i in range(len(confidence)):
                if i < window_size:
                    available_data = confidence[:i + 1]
                    missing_data_weight = (window_size - len(available_data)) / window_size
                    rate = (np.sum(
                        available_data) + missing_data_weight * window_size * overall_confidence) / window_size
                else:
                    rate = np.mean(confidence[i - window_size + 1:i + 1])
                confidence_rates.append(rate)

            non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                      if idx in session_result["non_m_trials"]])

            plotting_data = session_result['plotting_data']
            confidence_rates = np.array(confidence_rates)[non_m_indices]
            reward_outcomes = session_result["reward_outcomes"][non_m_indices]
            time_axis = session_result['time_axis']
            plot_title = f'LC Signal by Computer Confidence Quartiles: {subject_id} - {session_date}'

        except Exception as e:
            print(f"Error processing p-values for {subject_id}/{session_date}: {e}")
            return None

    # Create quartile bins based on all confidence values
    quartile_bins = pd.qcut(confidence_rates, q=4, labels=False)

    # Calculate average confidence for each quartile
    quartile_averages = []
    for quartile in range(4):
        quartile_avg = np.mean(confidence_rates[quartile_bins == quartile])
        quartile_averages.append(quartile_avg)
    
    # Print average confidence for each quartile
    print(f"\nAverage computer confidence by quartile:")
    for quartile in range(4):
        print(f"Quartile {quartile + 1}: {quartile_averages[quartile]:.4f}")

    # Create the plot
    plt.figure(figsize=(12, 7))
    # REVERSED color scheme compared to reward rate quartiles (red=Q1, blue=Q4)
    colors = ['red', 'orange', 'green', 'blue']  # From lowest to highest confidence
    
    # Track trial counts
    trial_counts = {
        'Q1': {'rewarded': 0, 'unrewarded': 0},
        'Q2': {'rewarded': 0, 'unrewarded': 0},
        'Q3': {'rewarded': 0, 'unrewarded': 0},
        'Q4': {'rewarded': 0, 'unrewarded': 0}
    }

    if win_loss:
        for quartile in range(4):
            quartile_rewarded = (quartile_bins == quartile) & (reward_outcomes == 1)
            quartile_unrewarded = (quartile_bins == quartile) & (reward_outcomes == 0)
            
            # Update trial counts
            trial_counts[f'Q{quartile + 1}']['rewarded'] = np.sum(quartile_rewarded)
            trial_counts[f'Q{quartile + 1}']['unrewarded'] = np.sum(quartile_unrewarded)

            if np.sum(quartile_rewarded) > 0:
                rewarded_avg = np.mean(plotting_data[quartile_rewarded], axis=0)
                rewarded_sem = calculate_sem(plotting_data[quartile_rewarded], axis=0)
                plt.fill_between(time_axis,
                                 rewarded_avg - rewarded_sem,
                                 rewarded_avg + rewarded_sem,
                                 color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, rewarded_avg,
                         color=colors[quartile], linewidth=2,
                         label=f'Q{quartile + 1}')

            if np.sum(quartile_unrewarded) > 0:
                unrewarded_avg = np.mean(plotting_data[quartile_unrewarded], axis=0)
                unrewarded_sem = calculate_sem(plotting_data[quartile_unrewarded], axis=0)
                plt.plot(time_axis, unrewarded_avg,
                         color=colors[quartile], linewidth=2, linestyle='--',
                         label=f'_Q{quartile + 1}')
    else:
        for quartile in range(4):
            quartile_trials = quartile_bins == quartile
            trial_count = np.sum(quartile_trials)
            
            # Calculate rewarded and unrewarded counts for this quartile
            rewarded_count = np.sum(quartile_trials & (reward_outcomes == 1))
            unrewarded_count = np.sum(quartile_trials & (reward_outcomes == 0))
            trial_counts[f'Q{quartile + 1}']['rewarded'] = rewarded_count
            trial_counts[f'Q{quartile + 1}']['unrewarded'] = unrewarded_count
            
            if trial_count > 0:
                quartile_avg = np.mean(plotting_data[quartile_trials], axis=0)
                quartile_sem = calculate_sem(plotting_data[quartile_trials], axis=0)

                plt.fill_between(time_axis,
                                 quartile_avg - quartile_sem,
                                 quartile_avg + quartile_sem,
                                 color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, quartile_avg,
                         color=colors[quartile], linewidth=2,
                         label=f'Q{quartile + 1}')

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    
    # Custom legend with integrated Low-High labels and arrow
    if win_loss:
        # Create a list of custom legend handles
        legend_handles = []
        
        # First add solid/dashed line explanation
        solid_line = plt.Line2D([0], [0], color='black', linewidth=2, label='Rewarded')
        dashed_line = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Unrewarded')
        legend_handles.extend([solid_line, dashed_line])
        
        # Add spacer
        legend_handles.append(plt.Line2D([0], [0], color='none', label=''))
        
        # Create colored lines for quartiles with Low/High labels - REVERSED order for computer confidence
        legend_handles.append(plt.Line2D([0], [0], color=colors[0], linewidth=2, label='Q1   Low Comp Conf'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[1], linewidth=2, label='Q2   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[2], linewidth=2, label='Q3   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[3], linewidth=2, label='Q4   High Comp Conf'))
        
        # Create the legend with all handles
        plt.legend(handles=legend_handles, loc='upper right', fontsize=12,
                  title="Computer Confidence", title_fontsize=12)
        
        # Add trial counts as text on the left side below the figure
        trial_count_text = "Trial counts:\n"
        for q in range(4):
            trial_count_text += f"Q{q+1}: {trial_counts[f'Q{q+1}']['rewarded']} rewarded, {trial_counts[f'Q{q+1}']['unrewarded']} unrewarded\n"
        plt.figtext(0.25, 0.02, trial_count_text, ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    else:
        # For non-win/loss plots, similar legend without the solid/dashed explanation
        legend_handles = []
        
        # Create colored lines for quartiles with Low/High labels - REVERSED order
        legend_handles.append(plt.Line2D([0], [0], color=colors[0], linewidth=2, label='Q1   Low Comp Conf'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[1], linewidth=2, label='Q2   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[2], linewidth=2, label='Q3   ↓'))
        legend_handles.append(plt.Line2D([0], [0], color=colors[3], linewidth=2, label='Q4   High Comp Conf'))
        
        # Create the legend
        plt.legend(handles=legend_handles, loc='upper right', fontsize=12,
                  title="Computer Confidence", title_fontsize=12)
        
        # Add trial counts as text on the left side below the figure
        trial_count_text = "Trial counts:\n"
        for q in range(4):
            trial_count_text += f"Q{q+1}: {trial_counts[f'Q{q+1}']['rewarded']} rewarded, {trial_counts[f'Q{q+1}']['unrewarded']} unrewarded\n"
        plt.figtext(0.25, 0.02, trial_count_text, ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add text with quartile averages at the right side bottom of the plot
    quartile_text = "Average confidence values:\n" + "\n".join([f"Q{q+1}: {avg:.4f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.75, 0.02, quartile_text, ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Make room for the text at the bottom

    # Save the figure
    fig_name = "computer_confidence_quartiles"
    if original_session_date is None:
        save_figure(plt.gcf(), subject_id, "pooled",
                f"{fig_name}{'_pooled'}{'_winloss' if win_loss else ''}")
    else:
        save_figure(plt.gcf(), subject_id, original_session_date,
                f"{fig_name}{'_winloss' if win_loss else ''}")

    plt.show()
    return {
        'quartile_bins': quartile_bins,
        'confidence_rates': confidence_rates,  
        'quartile_averages': quartile_averages
    }

def plot_choice_history(behavior_data, subject_id, session_date):
    # Extract choices and rewards
    choices = behavior_data['choice']
    rewards = behavior_data['reward']
    trials = np.arange(1, len(choices) + 1)
    
    # Create the figure
    fig = plt.figure(figsize=(12, 3))
    
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
    plt.title(f'Choice History: {subject_id} - {session_date}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_per_session_win_loss(subject_id, behavior_df=None):
    """
    Plot win/loss traces for each session of a subject with choice history
    and state probabilities

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet

    Returns:
    --------
    dict: Dictionary of session win-loss analyses
    """
    # Find all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)

    # Get matching pennies sessions from behavior dataframe if provided, otherwise load from parquet
    matching_pennies_sessions = set()
    subject_data = None

    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(
                f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                       if os.path.isdir(os.path.join(subject_path, d)) and
                       os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                       d in matching_pennies_sessions])

    # Find max peak for consistent y-axis scaling
    max_peak = float('-inf')
    min_peak = float('inf')
    session_analyses = {}
    valid_sessions = []

    # Check if state probability data is available
    has_state_data = False
    if subject_data is not None and 'p_stochastic' in subject_data.columns:
        # Check if at least one session has non-zero, non-NaN values
        has_state_data = subject_data['p_stochastic'].notna().any() and subject_data['p_stochastic'].any()

    # First pass: find valid sessions and determine y-axis scaling
    for session_date in sessions:
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue

        if len(session_result['non_m_trials']) < 100:
            print(
                f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue  # Skip this session

        # Add to valid sessions list since it has enough trials
        valid_sessions.append(session_date)

        # Filter out missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])

        # Get reward outcomes and photometry data for valid non-missed trials
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        session_plots = session_result['plotting_data']

        # Separate rewarded and unrewarded trials
        win_plots = session_plots[reward_outcomes == 1]
        loss_plots = session_plots[reward_outcomes == 0]

        # Skip this session if not enough win/loss trials
        if len(win_plots) < 5 or len(loss_plots) < 5:
            print(f"Skipping {subject_id}/{session_date}, not enough win/loss trials.")
            valid_sessions.pop()  # Remove from valid sessions
            continue

        # Calculate averages
        win_avg = np.mean(win_plots, axis=0)
        loss_avg = np.mean(loss_plots, axis=0)

        # Calculate SEMs
        win_sem = calculate_sem(win_plots, axis=0)
        loss_sem = calculate_sem(loss_plots, axis=0)

        # Find max/min for y-axis scaling
        max_peak = max(max_peak, np.max(win_avg + win_sem), np.max(loss_avg + loss_sem))
        min_peak = min(min_peak, np.min(win_avg - win_sem), np.min(loss_avg - loss_sem))

        # Store session data
        session_analyses[session_date] = {
            'win_avg': win_avg,
            'loss_avg': loss_avg,
            'win_sem': win_sem,
            'loss_sem': loss_sem,
            'win_count': len(win_plots),
            'loss_count': len(loss_plots),
            'session_result': session_result  # Keep the full result for later
        }

    if not valid_sessions:
        print(f"No valid sessions found for {subject_id}")
        return {}

    # Add some buffer to y-axis limits
    y_range = max_peak - min_peak
    y_max = max_peak + 0.05 * y_range
    y_min = min_peak - 0.05 * y_range

    # Calculate layout
    n_sessions = len(valid_sessions)
    n_cols = 3  # Show 3 sessions per row
    n_rows = (n_sessions + n_cols - 1) // n_cols  # Ceiling division

    # Determine number of rows per session (2 if no state data, 3 if state data exists)
    plots_per_session = 3 if has_state_data else 2

    # Create figure with proper size
    fig = plt.figure(figsize=(18, 5 * n_rows * plots_per_session / 3))  # Adjust height based on rows per session

    # Add a big general title above all plots
    fig.suptitle(f"Session History for {subject_id}", fontsize=24, y=0.98)

    # Create GridSpec to control the layout - 2 or 3 rows per session based on state data
    gs = plt.GridSpec(n_rows * plots_per_session, n_cols)

    # Second pass: create all the plots
    time_axis = None

    for i, session_date in enumerate(valid_sessions):
        row = i // n_cols
        col = i % n_cols

        session_data = session_analyses[session_date]
        session_result = session_data['session_result']
        behavior_data = session_result['behavioral_data']

        # Set time axis from the first valid session if not set yet
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Photometry plot (1st row of plots_per_session for this session)
        ax1 = fig.add_subplot(gs[row * plots_per_session, col])

        # Plot reward outcomes - using pre-calculated values from first pass
        ax1.fill_between(time_axis,
                         session_data['win_avg'] - session_data['win_sem'],
                         session_data['win_avg'] + session_data['win_sem'],
                         color='lightgreen', alpha=0.3)

        ax1.plot(time_axis, session_data['win_avg'], color='green', linewidth=2,
                 label=f'Win (n={session_data["win_count"]})')

        ax1.fill_between(time_axis,
                         session_data['loss_avg'] - session_data['loss_sem'],
                         session_data['loss_avg'] + session_data['loss_sem'],
                         color='lightsalmon', alpha=0.3)

        ax1.plot(time_axis, session_data['loss_avg'], color='darkorange', linewidth=2,
                 label=f'Loss (n={session_data["loss_count"]})')

        # Add vertical line at cue onset
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Lick Timing')

        # Use consistent y-axis scaling across all sessions
        ax1.set_ylim(y_min, y_max)

        # Add horizontal line at y=0
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Only add x labels to bottom row
        ax1.set_xlabel('Time (s)')

        # Only add y labels to leftmost column
        if col == 0:
            ax1.set_ylabel('ΔF/F')

        ax1.set_title(f"{session_date}", fontsize=12)

        # Add legend to first plot only
        if i == 0:
            ax1.legend(loc='upper right')

        # Choice history plot (2nd row of plots_per_session for this session)
        ax2 = fig.add_subplot(gs[row * plots_per_session + 1, col])

        # Extract choices and rewards
        choices = np.array(behavior_data['choice'])
        rewards = np.array(behavior_data['reward'])

        # Plot choice history
        for j, choice in enumerate(choices):
            if choice == 'L':
                ax2.plot([j + 1, j + 1], [0, 1], 'r-', linewidth=1.5)
                if rewards[j] == 1:
                    ax2.plot(j + 1, 1, 'ro', markersize=8, fillstyle='none')
            elif choice == 'R':
                ax2.plot([j + 1, j + 1], [0, -1], 'b-', linewidth=1.5)
                if rewards[j] == 1:
                    ax2.plot(j + 1, -1, 'bo', markersize=8, fillstyle='none')
            # 'M' choices result in a gap (no line)

        # Add the middle line
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)

        # Set the y-axis limits and labels
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Right', '', 'Left'])

        # Set the x-axis and title
        ax2.set_xlabel('Trial Number')
        ax2.set_title('Choice History')
        ax2.grid(True, alpha=0.3)

        # Only add state probabilities plot if data is available
        if has_state_data:
            # State probabilities plot (3rd row of plots_per_session for this session)
            ax3 = fig.add_subplot(gs[row * plots_per_session + 2, col])

            # Get state probability data for this session
            if behavior_df is not None:
                session_df = behavior_df[(behavior_df['subjid'] == subject_id) &
                                         (behavior_df['date'] == session_date)]
            else:
                # This should rarely happen since we already loaded the data above
                session_df = subject_data[subject_data['date'] == session_date]

            if not session_df.empty and 'p_stochastic' in session_df.columns:
                # Plot state probabilities if available
                x_values = np.arange(len(session_df))

                # Plot each state probability
                ax3.plot(x_values, session_df['p_stochastic'], color='green', linewidth=1.5, label='Stochastic')
                ax3.plot(x_values, session_df['p_leftbias'], color='red', linewidth=1.5, label='Left Bias')
                ax3.plot(x_values, session_df['p_rightbias'], color='blue', linewidth=1.5, label='Right Bias')

                # Add threshold line
                ax3.axhline(y=0.8, color='black', linestyle='--', linewidth=0.5, alpha=0.7, label='Threshold')

                # Add text marker for highest probability at final trial
                final_idx = len(session_df) - 1
                if final_idx >= 0:
                    probs = {
                        'Stochastic': session_df.iloc[final_idx]['p_stochastic'],
                        'Left Bias': session_df.iloc[final_idx]['p_leftbias'],
                        'Right Bias': session_df.iloc[final_idx]['p_rightbias']
                    }
                    max_state = max(probs, key=probs.get)
                    max_prob = probs[max_state]

                    if max_prob >= 0.8:  # Only add marker if probability is high enough
                        ax3.text(final_idx, max_prob, max_state[0],
                                 fontsize=8, ha='center', va='bottom')

                # Set y-axis limits
                ax3.set_ylim(-0.05, 1.05)
                ax3.set_ylabel('State Prob.')

                # Only show legend for first plot to avoid clutter
                if i == 0:
                    ax3.legend(loc='upper right', fontsize=7)

                # Set x-axis limits
                ax3.set_xlim(0, len(session_df))
            else:
                # No state data available
                ax3.text(0.5, 0.5, "No state probability data",
                         ha='center', va='center', transform=ax3.transAxes)
                ax3.set_ylabel('State Prob.')
                ax3.set_ylim(0, 1)

            ax3.set_xlabel('Trial Number')

    # Adjust layout to make room for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for suptitle
    # Save the figure
    save_figure(fig, subject_id, "all_sessions", "per_session_win_loss_with_choices_and_states")
    plt.show()

def analyze_session_win_loss_difference_gap(subject_id, session_date=None, comp_conf=False, behavior_df=None, sem=True):
    """
    Analyze win-loss difference across photometry sessions for a subject

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    comp_conf : bool, optional
        If True, sorts sessions by computer confidence rather than chronologically
    df : pandas.DataFrame, optional
        Pre-loaded dataframe to use for confidence calculations

    Returns:
    --------
    dict: Dictionary of session win-loss difference analyses
    """
    if session_date is None:
        if behavior_df is not None:
            # Get sessions from the provided dataframe
            sessions = sorted(behavior_df[behavior_df['subjid'] == subject_id]['date'].unique())
        else:
            # Get sessions from the filesystem
            subject_path = os.path.join(base_dir, subject_id)
            sessions = sorted([d for d in os.listdir(subject_path)
                               if os.path.isdir(os.path.join(subject_path, d)) and
                               os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])
    else:
        sessions = [session_date]

    # Load dataframe if needed for confidence calculations and not provided
    if comp_conf and behavior_df is None:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting
    else:
        df = behavior_df  

    # Store results for each session
    session_differences = {}
    session_confidences = {}

    # Process each session
    for idx, session_date in enumerate(sessions):
        # Process the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df, z_score=False)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue

        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        # Filter out missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"]) 
                                  if idx in session_result["non_m_trials"]])

        # Get reward outcomes and photometry data
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        session_plots = session_result['plotting_data']

        # Separate rewarded and unrewarded trials
        rewarded_trials = session_plots[reward_outcomes == 1]
        unrewarded_trials = session_plots[reward_outcomes == 0]

        # Compute average rewarded and unrewarded signals with SEM
        rewarded_avg = np.mean(rewarded_trials, axis=0)
        unrewarded_avg = np.mean(unrewarded_trials, axis=0)
        rewarded_sem = calculate_sem(rewarded_trials, axis=0)
        unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)

        # Compute win-loss difference
        rewarded_avg = rewarded_avg + np.abs(np.min([rewarded_avg, unrewarded_avg]))
        unrewarded_avg = unrewarded_avg + np.abs(np.min([rewarded_avg, unrewarded_avg]))
        win_loss_diff = rewarded_avg - unrewarded_avg
        win_loss_sem = np.sqrt(rewarded_sem**2 + unrewarded_sem**2)

        # Store the difference data
        session_differences[session_date] = {
            'diff': win_loss_diff,
            'sem': win_loss_sem,
            'time_axis': session_result['time_axis']
        }

        # Calculate computer confidence if requested
        if comp_conf:
            try:
                # Get data for this session from provided or loaded dataframe
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
                
                if session_df.empty:
                    print(f"No behavioral data found for {subject_id}/{session_date}")
                    continue
                
                # Extract p-values and calculate confidence
                if 'min_pvalue' in session_df.columns:
                    # Extract p-values
                    p_values = session_df['min_pvalue'].values
                    
                    # Remove NaN values
                    p_values = p_values[~np.isnan(p_values)]
                    
                    if len(p_values) == 0:
                        print(f"No valid p-values for {subject_id}/{session_date}")
                        continue
                    
                    # Cap very small p-values at 10^-12 to avoid infinite confidence
                    min_p_value = 1e-12
                    p_values = np.maximum(p_values, min_p_value)
                    
                    # Calculate confidence as -log10(p_value)
                    confidence = -np.log10(p_values)
                    
                    # Calculate average confidence for the session
                    avg_confidence = np.mean(confidence)
                    session_confidences[session_date] = avg_confidence
                    print(f"Session {session_date} average confidence: {avg_confidence:.4f}")
                else:
                    print(f"No min_pvalue column found in data for {subject_id}/{session_date}")
            except Exception as e:
                print(f"Error calculating confidence for {subject_id}/{session_date}: {e}")

    # Check if we have any valid sessions
    if not session_differences:
        print(f"No valid sessions found for {subject_id}")
        return None

    # Sort sessions based on confidence or chronologically
    if comp_conf and session_confidences:
        # Filter out sessions with missing confidence values
        valid_sessions = [s for s in session_confidences.keys() if s in session_differences]
        
        if not valid_sessions:
            print("No sessions with valid confidence values found")
            sorted_sessions = sorted(session_differences.keys())
        else:
            # Sort by confidence (highest first)
            sorted_sessions = sorted(valid_sessions, key=lambda s: session_confidences.get(s, 0), reverse=True)
            
            # Print the confidence ranking
            print("\nSessions ranked by computer confidence (highest to lowest):")
            for i, sess in enumerate(sorted_sessions):
                print(f"{i+1}. Session {sess}: {session_confidences[sess]:.4f}")
    else:
        # Use chronological sorting (default)
        sorted_sessions = sorted(session_differences.keys())

    # Create colors for plotting
    num_sessions = len(sorted_sessions)
    
    # Special coloring for JOA-M-0020: Yellow -> Green -> Blue gradient (viridis-like)
    if subject_id == "JOA-M-0020":
        if comp_conf and session_confidences:
            # When sorting by confidence, use viridis colormap (yellow -> green -> blue -> purple)
            colors = plt.cm.viridis(np.linspace(0, 1, num_sessions))
        else:
            # For chronological sorting, also use viridis for JOA-M-0020
            colors = plt.cm.viridis(np.linspace(0, 1, num_sessions))
    else:
        # Regular Blues colormap for other subjects
        if comp_conf and session_confidences:
            # When sorting by confidence, use reversed color gradient (highest confidence = lightest blue)
            colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))
        else:
            # Default: earliest session = lightest blue
            colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))

    # Create plot
    plt.figure(figsize=(12, 7))
    chronological_sessions = sorted(session_differences.keys())

    # Create mapping from session date to chronological position (1-indexed)
    chronological_order = {sess_date: idx+1 for idx, sess_date in enumerate(chronological_sessions)}

    # Plot each session's win-loss difference
    for idx, sess_date in enumerate(sorted_sessions):
        time_axis = session_differences[sess_date]['time_axis']
        win_loss_diff = session_differences[sess_date]['diff']
        win_loss_sem = session_differences[sess_date]['sem']
        
        # Create label with confidence value if available
        if comp_conf and sess_date in session_confidences:
            label = f'Session {chronological_order[sess_date]} (conf: {session_confidences[sess_date]:.2f})'
        else:
            label = f'Session {chronological_order[sess_date]}'
        
        # Plot with shaded error region
        if sem:
            plt.fill_between(time_axis,
                            win_loss_diff - win_loss_sem,
                            win_loss_diff + win_loss_sem,
                            color=colors[idx], alpha=0.2)
        plt.plot(time_axis, win_loss_diff,
                color=colors[idx],
             label=label, linewidth=2)

    # Add reference lines
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels and formatting
    plt.xlabel('Time (s) after first lick', fontsize=24)
    plt.ylabel('Rewarded - Unrewarded ΔF/F', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    sort_type = "by computer confidence" if comp_conf else "chronologically"
    plt.title(f'Win-Loss Difference: {subject_id} (sorted {sort_type})', fontsize=26)
    
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the figure
    sort_suffix = "by_comp_conf" if comp_conf else "chronological"
    sem_suffix = "_sem" if sem else ""
    save_figure(plt.gcf(), subject_id, "win_loss_diff", f"win_loss_difference_{sort_suffix}{sem_suffix}")

    plt.show()

    # Return results
    return {
        'session_differences': session_differences,
        'session_confidences': session_confidences if comp_conf else None,
        'sorted_sessions': sorted_sessions
    }

def analyze_previous_outcome_effect(subject_id="All", time_split=False, behavior_df=None, specific_subjects=None):
    """
    Analyze photometry signals based on previous and current trial outcomes.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject or "All" for cross-subject analysis
    time_split : bool, optional (default=False)
        If True, additionally split data by early/middle/late sessions
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    specific_subjects : list, optional
        List of subject IDs to include if subject_id="All"

    Returns:
    --------
    dict: Analysis results
    """
    # Handle cross-subject analysis
    if subject_id == "All":
        if specific_subjects is None:
            # Default list of subjects
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Store subject-level averages for each condition
        subject_condition_avgs = {
            'prev_win_curr_win': [],
            'prev_win_curr_loss': [],
            'prev_loss_curr_win': [],
            'prev_loss_curr_loss': []
        }
        
        time_axis = None
        total_sessions = 0
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for previous outcome effect analysis...")
            
            # Process the individual subject (not doing time_split for cross-subject analysis)
            subj_result = analyze_previous_outcome_effect_single(subj, False, behavior_df)
            
            if subj_result and 'condition_data' in subj_result:
                if time_axis is None:
                    time_axis = subj_result['time_axis']
                    
                total_sessions += subj_result['num_sessions']
                
                # Add each condition's average to the subject-level collection
                for condition in ['prev_win_curr_win', 'prev_win_curr_loss', 'prev_loss_curr_win', 'prev_loss_curr_loss']:
                    if subj_result['condition_data'][condition]['avg'] is not None:
                        subject_condition_avgs[condition].append(subj_result['condition_data'][condition]['avg'])
        
        # Create the plot if we have data
        if time_axis is None or not any(subject_condition_avgs.values()):
            print("No valid data found for cross-subject analysis")
            return None
            
        plt.figure(figsize=(12, 7))
        
        # Define colors and labels
        colors = {
            'prev_win_curr_win': '#117733',  # green
            'prev_win_curr_loss': '#DDCC77', # yellow
            'prev_loss_curr_win': '#4477AA', # blue
            'prev_loss_curr_loss': '#CC6677' # red
        }
        
        # Calculate and plot the cross-subject average for each condition
        for condition, color in colors.items():
            if len(subject_condition_avgs[condition]) > 0:
                # Calculate mean and SEM across subjects
                condition_mean = np.mean(subject_condition_avgs[condition], axis=0)
                condition_sem = np.std(subject_condition_avgs[condition], axis=0) / np.sqrt(len(subject_condition_avgs[condition]))
                
                labels = {
                    "prev_win_curr_win": "Win → Win", 
                    "prev_win_curr_loss": "Win → Loss",
                    "prev_loss_curr_win": "Loss → Win",
                    "prev_loss_curr_loss": "Loss → Loss"
                    }
                
                plt.fill_between(time_axis,
                               condition_mean - condition_sem,
                               condition_mean + condition_sem,
                               color=color, alpha=0.3)
                plt.plot(time_axis, condition_mean,
                       color=color, linewidth=2, 
                       label=f'{labels[condition]}')
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Labels and formatting
        plt.xlabel('Time (s) after first lick', fontsize=24)
        plt.ylabel('z-ΔF/F', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title(f'LC Signal by Previous Trial Outcome: All Subjects (n={len(specific_subjects)})', 
                 fontsize=26)
        plt.xlim([-pre_cue_time, post_cue_time])
        plt.legend(loc='upper right', fontsize=24)
        plt.tight_layout()
        
        # Save the figure
        save_figure(plt.gcf(), "all_subjects", "pooled", "previous_outcome_effect")
        plt.show()
        
        # Return result
        return {
            'subject_id': 'All',
            'specific_subjects': specific_subjects,
            'subject_condition_avgs': subject_condition_avgs,
            'time_axis': time_axis,
            'total_sessions': total_sessions
        }
    
    else:
        # Original single-subject behavior
        return analyze_previous_outcome_effect_single(subject_id, time_split, behavior_df)

def analyze_previous_outcome_effect_single(subject_id, time_split=False, behavior_df=None):
    """
    Analyze photometry signals based on previous and current trial outcomes.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    time_split : bool, optional (default=False)
        If True, additionally split data by early/middle/late sessions and
        create separate plots showing temporal evolution
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet

    Returns:
    --------
    dict: Analysis results
    """

    subject_path = os.path.join(base_dir, subject_id)

    # Find all session directories for this subject
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(
                f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                       if os.path.isdir(os.path.join(subject_path, d)) and
                       os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                       d in matching_pennies_sessions])

    # Process each session and collect results
    all_sessions = []
    all_plotting_data = []
    all_prev_rewards = []
    all_curr_rewards = []
    session_dates = []

    # For time split analysis, we'll store data separately for each time period
    if time_split:
        # First pass: identify valid sessions with enough trials
        valid_sessions = []

        # Check each session for having enough trials
        for session_date in sessions:
            session_path = os.path.join(subject_path, session_date)
            if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
                # Process session to check trial count
                result = process_session(subject_id, session_date, behavior_df=behavior_df)
                if result and len(result['non_m_trials']) >= 100:
                    valid_sessions.append(session_date)
                else:
                    # Use 'result' instead of 'session_result' to avoid the NameError
                    if result:
                        print(
                            f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(result['non_m_trials'])}).")
                    else:
                        print(f"Skipping {subject_id}/{session_date}, could not process session.")

        # Now split the valid sessions evenly
        num_valid = len(valid_sessions)
        if num_valid < 3:
            print(f"Warning: Only {num_valid} valid sessions available. Need at least 3 for time split analysis.")
            time_split = False  # Fall back to non-split analysis
        else:
            # Calculate split points for even division
            early_end = num_valid // 3
            middle_end = 2 * (num_valid // 3)

            # Adjust for remainder (ensure most even split possible)
            if num_valid % 3 == 1:
                # Add the extra session to the last group
                early_sessions = valid_sessions[:early_end]
                middle_sessions = valid_sessions[early_end:middle_end]
                late_sessions = valid_sessions[middle_end:]
            elif num_valid % 3 == 2:
                # Add one extra session to middle and late groups
                early_sessions = valid_sessions[:early_end]
                middle_sessions = valid_sessions[early_end:middle_end + 1]
                late_sessions = valid_sessions[middle_end + 1:]
            else:
                # Perfect division
                early_sessions = valid_sessions[:early_end]
                middle_sessions = valid_sessions[early_end:middle_end]
                late_sessions = valid_sessions[middle_end:]

            # Initialize data containers for each time period
            time_periods = {
                'early': {'sessions': [], 'plotting_data': [], 'prev_rewards': [], 'curr_rewards': []},
                'middle': {'sessions': [], 'plotting_data': [], 'prev_rewards': [], 'curr_rewards': []},
                'late': {'sessions': [], 'plotting_data': [], 'prev_rewards': [], 'curr_rewards': []}
            }
            
            print(f"Time split analysis: Early ({len(early_sessions)} sessions), " +
                  f"Middle ({len(middle_sessions)} sessions), Late ({len(late_sessions)} sessions)")

    # Process sessions in chronological order
    for idx, session_date in enumerate(sessions):
        session_path = os.path.join(subject_path, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            # Pass behavior_df to process_session to reuse data
            result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if result:
                if len(result['non_m_trials']) < 100:
                    print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(result['non_m_trials'])}).")
                    continue

                all_sessions.append(result)
                session_dates.append(session_date)

                # Get behavioral data
                behavior_data = result['behavioral_data']
                rewards = behavior_data['reward']

                # Filter out missed trials
                non_m_indices = np.array([i for i, idx in enumerate(result["valid_trials"])
                                          if idx in result["non_m_trials"]])

                # Extract data for valid trials
                session_plots = result['plotting_data']
                curr_rewards = result["reward_outcomes"][non_m_indices]

                # Create previous reward array (shifted)
                # First trial has no previous trial, assign -1 (will be filtered out)
                prev_rewards = np.zeros_like(curr_rewards)
                prev_rewards[0] = -1  # No previous trial for first trial
                prev_rewards[1:] = curr_rewards[:-1]

                # Store data for overall analysis
                all_plotting_data.append(session_plots)
                all_curr_rewards.append(curr_rewards)
                all_prev_rewards.append(prev_rewards)

                # If doing time split analysis, store in the appropriate time period
                if time_split:
                    # Determine which period this session belongs to
                    if session_date in early_sessions:
                        period = 'early'
                    elif session_date in middle_sessions:
                        period = 'middle'
                    elif session_date in late_sessions:
                        period = 'late'
                    else:
                        # Skip sessions that were excluded during validation
                        continue
                        
                    time_periods[period]['sessions'].append(result)
                    time_periods[period]['plotting_data'].append(session_plots)
                    time_periods[period]['curr_rewards'].append(curr_rewards)
                    time_periods[period]['prev_rewards'].append(prev_rewards)
    

    if not all_sessions:
        print(f"No processed sessions found for subject {subject_id}")
        return None

    # Combine all trials from all sessions
    plotting_data = np.vstack(all_plotting_data)
    curr_rewards = np.concatenate(all_curr_rewards)
    prev_rewards = np.concatenate(all_prev_rewards)

    # Filter out first trials (no previous outcome)
    valid_trials = prev_rewards != -1
    plotting_data = plotting_data[valid_trials]
    curr_rewards = curr_rewards[valid_trials]
    prev_rewards = prev_rewards[valid_trials]

    # Get time axis
    time_axis = all_sessions[0]['time_axis']

    # Create condition masks
    prev_win_curr_win = (prev_rewards == 1) & (curr_rewards == 1)
    prev_win_curr_loss = (prev_rewards == 1) & (curr_rewards == 0)
    prev_loss_curr_win = (prev_rewards == 0) & (curr_rewards == 1)
    prev_loss_curr_loss = (prev_rewards == 0) & (curr_rewards == 0)

    # Calculate averages and SEM for each condition
    condition_data = {
        'prev_win_curr_win': {
            'data': plotting_data[prev_win_curr_win],
            'avg': np.mean(plotting_data[prev_win_curr_win], axis=0) if np.any(prev_win_curr_win) else None,
            'sem': calculate_sem(plotting_data[prev_win_curr_win], axis=0) if np.any(prev_win_curr_win) else None,
            'count': np.sum(prev_win_curr_win)
        },
        'prev_win_curr_loss': {
            'data': plotting_data[prev_win_curr_loss],
            'avg': np.mean(plotting_data[prev_win_curr_loss], axis=0) if np.any(prev_win_curr_loss) else None,
            'sem': calculate_sem(plotting_data[prev_win_curr_loss], axis=0) if np.any(prev_win_curr_loss) else None,
            'count': np.sum(prev_win_curr_loss)
        },
        'prev_loss_curr_win': {
            'data': plotting_data[prev_loss_curr_win],
            'avg': np.mean(plotting_data[prev_loss_curr_win], axis=0) if np.any(prev_loss_curr_win) else None,
            'sem': calculate_sem(plotting_data[prev_loss_curr_win], axis=0) if np.any(prev_loss_curr_win) else None,
            'count': np.sum(prev_loss_curr_win)
        },
        'prev_loss_curr_loss': {
            'data': plotting_data[prev_loss_curr_loss],
            'avg': np.mean(plotting_data[prev_loss_curr_loss], axis=0) if np.any(prev_loss_curr_loss) else None,
            'sem': calculate_sem(plotting_data[prev_loss_curr_loss], axis=0) if np.any(prev_loss_curr_loss) else None,
            'count': np.sum(prev_loss_curr_loss)
        }
    }

    # If doing time split analysis, compute condition data for each time period
    time_period_data = {}
    if time_split:
        for period in ['early', 'middle', 'late']:
            if not time_periods[period]['sessions']:
                print(f"No sessions in {period} period. Skipping.")
                continue

            # Combine data for this time period
            period_plotting_data = np.vstack(time_periods[period]['plotting_data'])
            period_curr_rewards = np.concatenate(time_periods[period]['curr_rewards'])
            period_prev_rewards = np.concatenate(time_periods[period]['prev_rewards'])

            # Filter out first trials
            period_valid = period_prev_rewards != -1
            period_plotting_data = period_plotting_data[period_valid]
            period_curr_rewards = period_curr_rewards[period_valid]
            period_prev_rewards = period_prev_rewards[period_valid]

            # Create condition masks
            period_prev_win_curr_win = (period_prev_rewards == 1) & (period_curr_rewards == 1)
            period_prev_win_curr_loss = (period_prev_rewards == 1) & (period_curr_rewards == 0)
            period_prev_loss_curr_win = (period_prev_rewards == 0) & (period_curr_rewards == 1)
            period_prev_loss_curr_loss = (period_prev_rewards == 0) & (period_curr_rewards == 0)

            # Calculate averages and SEM for each condition
            time_period_data[period] = {
                'prev_win_curr_win': {
                    'data': period_plotting_data[period_prev_win_curr_win],
                    'avg': np.mean(period_plotting_data[period_prev_win_curr_win], axis=0)
                    if np.any(period_prev_win_curr_win) else None,
                    'sem': calculate_sem(period_plotting_data[period_prev_win_curr_win], axis=0)
                    if np.any(period_prev_win_curr_win) else None,
                    'count': np.sum(period_prev_win_curr_win)
                },
                'prev_win_curr_loss': {
                    'data': period_plotting_data[period_prev_win_curr_loss],
                    'avg': np.mean(period_plotting_data[period_prev_win_curr_loss], axis=0)
                    if np.any(period_prev_win_curr_loss) else None,
                    'sem': calculate_sem(period_plotting_data[period_prev_win_curr_loss], axis=0)
                    if np.any(period_prev_win_curr_loss) else None,
                    'count': np.sum(period_prev_win_curr_loss)
                },
                'prev_loss_curr_win': {
                    'data': period_plotting_data[period_prev_loss_curr_win],
                    'avg': np.mean(period_plotting_data[period_prev_loss_curr_win], axis=0)
                    if np.any(period_prev_loss_curr_win) else None,
                    'sem': calculate_sem(period_plotting_data[period_prev_loss_curr_win], axis=0)
                    if np.any(period_prev_loss_curr_win) else None,
                    'count': np.sum(period_prev_loss_curr_win)
                },
                'prev_loss_curr_loss': {
                    'data': period_plotting_data[period_prev_loss_curr_loss],
                    'avg': np.mean(period_plotting_data[period_prev_loss_curr_loss], axis=0)
                    if np.any(period_prev_loss_curr_loss) else None,
                    'sem': calculate_sem(period_plotting_data[period_prev_loss_curr_loss], axis=0)
                    if np.any(period_prev_loss_curr_loss) else None,
                    'count': np.sum(period_prev_loss_curr_loss)
                }
            }

    # Create the regular (non-time-split) plot
    plt.figure(figsize=(12, 7))

    # Define colors and labels
    colors = {
        'prev_win_curr_win': '#117733',  # green
        'prev_win_curr_loss': '#DDCC77', # yellow
        'prev_loss_curr_win': '#4477AA', # blue
        'prev_loss_curr_loss': '#CC6677' # red
    }

    labels = {
        'prev_win_curr_win': f"Win-Win (n={condition_data['prev_win_curr_win']['count']})",
        'prev_win_curr_loss': f"Win-Loss (n={condition_data['prev_win_curr_loss']['count']})",
        'prev_loss_curr_win': f"Loss-Win (n={condition_data['prev_loss_curr_win']['count']})",
        'prev_loss_curr_loss': f"Loss-Loss (n={condition_data['prev_loss_curr_loss']['count']})"
    }

    # Plot each condition
    for condition, color in colors.items():
        if condition_data[condition]['avg'] is not None:
            plt.fill_between(time_axis,
                             condition_data[condition]['avg'] - condition_data[condition]['sem'],
                             condition_data[condition]['avg'] + condition_data[condition]['sem'],
                             color=color, alpha=0.3)
            plt.plot(time_axis, condition_data[condition]['avg'],
                     color=color, linewidth=2, label=labels[condition])

    # Add vertical line at cue onset
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels and formatting
    plt.xlabel('Time (s) after first lick', fontsize=24)
    plt.ylabel('z-ΔF/F', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title(f'LC Signal by Previous Trial Outcome: {subject_id}', fontsize=26)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right', fontsize=24)
    plt.tight_layout()

    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", "previous_outcome_effect")
    plt.show()

    # If time_split is enabled, create additional plots
    if time_split:
        # Define style parameters for time-split plots
        period_colors = {'early': 'lightskyblue', 'middle': 'royalblue', 'late': 'darkblue'}
        condition_styles = {
            'prev_win_curr_win': {'color': '#117733', 'linestyle': '-', 'marker': 'o'},
            'prev_win_curr_loss': {'color': '#DDCC77', 'linestyle': '-', 'marker': 's'},
            'prev_loss_curr_win': {'color': '#4477AA', 'linestyle': '-', 'marker': '^'},
            'prev_loss_curr_loss': {'color': '#CC6677', 'linestyle': '-', 'marker': 'D'}
        }

        # 1. Create plots for each time period (early, middle, late) - all conditions
        for period in ['early', 'middle', 'late']:
            if period not in time_period_data:
                continue

            plt.figure(figsize=(12, 7))

            # Plot each condition for this time period
            for condition, style in condition_styles.items():
                if time_period_data[period][condition]['avg'] is not None:
                    count = time_period_data[period][condition]['count']
                    plt.fill_between(time_axis,
                                     time_period_data[period][condition]['avg'] - time_period_data[period][condition][
                                         'sem'],
                                     time_period_data[period][condition]['avg'] + time_period_data[period][condition][
                                         'sem'],
                                     color=style['color'], alpha=0.3)
                    plt.plot(time_axis, time_period_data[period][condition]['avg'],
                            color=style['color'], linewidth=2,
                            label=f"{condition.replace('prev_', '').replace('curr_', '-').replace('_', '').title()} (n={count})")

            # Add vertical line at cue onset
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Labels and formatting
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('ΔF/F', fontsize=16)

            # Count sessions in this period
            num_sessions = len(time_periods[period]['sessions'])
            plt.title(
                f'{period.capitalize()} Sessions - Previous Outcome Effect: {subject_id} ({num_sessions} sessions)',
                fontsize=20)

            plt.xlim([-pre_cue_time, post_cue_time])
            plt.legend(loc='upper right', fontsize=16)
            plt.tight_layout()

            # Save the figure
            save_figure(plt.gcf(), subject_id, "pooled", f"previous_outcome_effect_{period}")
            plt.show()

        # 2. Create plots for each condition (across time periods)
        for condition, style in condition_styles.items():
            plt.figure(figsize=(12, 7))

            # Plot this condition for each time period
            for period, color in period_colors.items():
                if period in time_period_data and time_period_data[period][condition]['avg'] is not None:
                    count = time_period_data[period][condition]['count']
                    plt.fill_between(time_axis,
                                     time_period_data[period][condition]['avg'] - time_period_data[period][condition][
                                         'sem'],
                                     time_period_data[period][condition]['avg'] + time_period_data[period][condition][
                                         'sem'],
                                     color=color, alpha=0.3)
                    plt.plot(time_axis, time_period_data[period][condition]['avg'],
                             color=color, linewidth=2,
                             label=f"{period.capitalize()} Sessions (n={count})")

            # Add vertical line at cue onset
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Labels and formatting
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('ΔF/F', fontsize=16)

            # Format condition name for title
            condition_title = condition.replace('prev_', '').replace('curr_', '-').replace('_', '').title()
            plt.title(f'Temporal Evolution of {condition_title}: {subject_id}', fontsize=20)

            plt.xlim([-pre_cue_time, post_cue_time])
            plt.legend(loc='upper right', fontsize=16)
            plt.tight_layout()

            # Save the figure
            save_figure(plt.gcf(), subject_id, "pooled", f"previous_outcome_effect_evolution_{condition}")
            plt.show()

    # Return analysis results
    result = {
        'subject_id': subject_id,
        'condition_data': condition_data,
        'time_axis': time_axis,
        'num_sessions': len(all_sessions),
        'session_dates': session_dates
    }
    return result


def analyze_selected_previous_outcome(subject_id, selected_conditions=None, behavior_df=None):
    """
    Analyze photometry signals for selected previous and current outcome combinations.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    selected_conditions : list or str, optional
        Specific outcome combinations to show. Can be a single string or list of strings.
        Options: "win_win", "win_loss", "loss_win", "loss_loss", or "all"
        If None or "all", all four combinations will be shown
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
        
    Returns:
    --------
    dict: Analysis results for the selected outcome combinations
    """
    # Validate and standardize selected_conditions
    valid_conditions = ["win_win", "win_loss", "loss_win", "loss_loss"]
    
    if selected_conditions is None or selected_conditions == "all":
        selected_conditions = valid_conditions
    elif isinstance(selected_conditions, str):
        if selected_conditions in valid_conditions:
            selected_conditions = [selected_conditions]
        else:
            print(f"Invalid condition '{selected_conditions}'. Using all conditions.")
            selected_conditions = valid_conditions
    else:
        # Filter out any invalid conditions
        selected_conditions = [cond for cond in selected_conditions if cond in valid_conditions]
        if not selected_conditions:
            print("No valid conditions provided. Using all conditions.")
            selected_conditions = valid_conditions
            
    print(f"Analyzing {len(selected_conditions)} outcome combinations: {', '.join(selected_conditions)}")
    
    # Find all session directories for this subject
    subject_path = os.path.join(base_dir, subject_id)
    matching_pennies_sessions = set()
    
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_path)
                      if os.path.isdir(os.path.join(subject_path, d)) and
                      os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                      d in matching_pennies_sessions])
    
    # Collect data by previous and current outcome
    outcome_data = {
        "win_win": [], 
        "win_loss": [],
        "loss_win": [],
        "loss_loss": []
    }
    
    time_axis = None
    total_sessions = 0
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        
        # Process the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue
        
        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue
            
        total_sessions += 1
        
        # Get time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']
        
        # Get rewards
        rewards = np.array(session_result['behavioral_data']['reward'])
        
        # Map from non_m_trials (indices in plotting_data) back to original trial indices
        valid_trials = np.array(session_result["valid_trials"])
        non_m_trials = np.array(session_result["non_m_trials"])
        
        # Process each non-missed trial except the first one
        for i in range(1, len(non_m_trials)):
            # Get the actual trial indices
            curr_trial_idx = non_m_trials[i]
            prev_trial_idx = non_m_trials[i-1]
            
            # Skip if these trials are not consecutive
            if prev_trial_idx + 1 != curr_trial_idx:
                continue
                
            # Get outcomes
            prev_outcome = rewards[prev_trial_idx]
            curr_outcome = rewards[curr_trial_idx]
                
            # Previous win, current win
            if prev_outcome == 1 and curr_outcome == 1:
                outcome_data["win_win"].append(session_result['plotting_data'][i])
            
            # Previous win, current loss
            elif prev_outcome == 1 and curr_outcome == 0:
                outcome_data["win_loss"].append(session_result['plotting_data'][i])
            
            # Previous loss, current win
            elif prev_outcome == 0 and curr_outcome == 1:
                outcome_data["loss_win"].append(session_result['plotting_data'][i])
            
            # Previous loss, current loss
            elif prev_outcome == 0 and curr_outcome == 0:
                outcome_data["loss_loss"].append(session_result['plotting_data'][i])
    
    if total_sessions == 0:
        print(f"No valid sessions found for {subject_id}")
        return None
        
    # Calculate averages and SEMs for each condition
    for condition in outcome_data:
        if outcome_data[condition]:
            outcome_data[condition] = np.array(outcome_data[condition])
    
    # Plot selected conditions
    plt.figure(figsize=(12, 7))
    
    # Colors for different conditions
    colors = {
        "win_win": '#117733', 
        "win_loss": '#DDCC77',
        "loss_win": '#4477AA',
        "loss_loss": '#CC6677'
    }
    
    # Labels for different conditions
    labels = {
        "win_win": "Win → Win", 
        "win_loss": "Win → Loss",
        "loss_win": "Loss → Win",
        "loss_loss": "Loss → Loss"
    }
    
    # Plot each selected condition
    for condition in selected_conditions:
        if condition in outcome_data and len(outcome_data[condition]) > 0:
            condition_avg = np.mean(outcome_data[condition], axis=0)
            condition_sem = calculate_sem(outcome_data[condition], axis=0)
            
            plt.fill_between(time_axis, 
                           condition_avg - condition_sem,  
                           condition_avg + condition_sem,  
                           color=colors[condition], alpha=0.3)
            
            plt.plot(time_axis, condition_avg,
                   color=colors[condition], linewidth=2.5,
                   label=f'{labels[condition]}')
    
    # Add vertical line at cue onset
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Labels and title
    plt.xlabel('Time (s) after first lick', fontsize=24)
    plt.ylabel('z-ΔF/F', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'Previous and Current Outcome Effects: {subject_id}', fontsize=26)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right', fontsize=24)
    plt.tight_layout()
    
    # Save the figure
    condition_label = '_'.join(selected_conditions) if len(selected_conditions) <= 2 else 'selected'
    save_figure(plt.gcf(), subject_id, "pooled", f"previous_outcome_effect_{condition_label}")
    
    plt.show()
    
    # Return the data
    return {
        'subject_id': subject_id,
        'total_sessions': total_sessions,
        'selected_conditions': selected_conditions,
        'outcome_data': outcome_data,
        'time_axis': time_axis
    }

def analyze_win_stay_lose_stay(subject_id, session_date=None, behavior_df=None):
    """
    Calculate Win-Stay, Lose-Stay statistics for a subject using all behavioral trials.
    This measures perseverance (tendency to repeat choices) regardless of outcome.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file

    Returns:
    --------
    dict: Analysis results including stay counts and percentages after wins and losses
    """
    if behavior_df is not None:
        df = behavior_df
        if 'date' in df.columns and df['date'].dtype != str:
            df['date'] = df['date'].astype(str)
    else:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            df = df[df['protocol'].str.contains('MatchingPennies', na=False)]
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            df = None  # We'll proceed without state analysis if file can't be loaded

    # Store results
    perseverance_results = {
        'win_stay_count': 0,
        'lose_stay_count': 0,
        'total_win_trials': 0,
        'total_lose_trials': 0,
        'sessions_analyzed': 0
    }
    
    # Store per-session results
    session_data = []

    # Get sessions to analyze
    if session_date is None:
        # Get all sessions from provided dataframe for this subject
        sessions = sorted(df[df['subjid'] == subject_id]['date'].unique())
    else:
        sessions = [session_date]

    # Process each session
    for sess in sessions:
        print(f"Analyzing perseverance for {subject_id}/{sess}...")
        
        # Get behavior data directly from parquet file
        try:
            session_df = df[(df['subjid'] == subject_id) & (df['date'] == sess)]
            
            if session_df.empty:
                print(f"No behavioral data found for {subject_id}/{sess}")
                continue
                
            choices = session_df['choice'].tolist()
            rewards = session_df['reward'].tolist()
        except Exception as e:
            print(f"Error loading behavioral data for {subject_id}/{sess}: {e}")
            continue

        # Filter out missed ('M') trials
        valid_indices = [i for i, choice in enumerate(choices) if choice in ['L', 'R']]
        valid_choices = [choices[i] for i in valid_indices]
        valid_rewards = [rewards[i] for i in valid_indices]

        # Skip session if not enough valid trials
        if len(valid_choices) < 2:
            print(f"Session {sess} has fewer than 2 valid trials. Skipping...")
            continue

        # Count stay behaviors after wins and losses
        win_stay_count = 0
        lose_stay_count = 0
        win_trials = 0
        lose_trials = 0

        # We start from the second trial since we need to know the previous trial's outcome
        for i in range(1, len(valid_choices)):
            prev_choice = valid_choices[i - 1]
            curr_choice = valid_choices[i]
            prev_reward = valid_rewards[i - 1]
            stayed = prev_choice == curr_choice

            # Count win and lose trials from previous trial
            if prev_reward == 1:  # Win
                win_trials += 1
                if stayed:
                    win_stay_count += 1
            else:  # Loss
                lose_trials += 1
                if stayed:
                    lose_stay_count += 1

        # Add to total counts
        perseverance_results['win_stay_count'] += win_stay_count
        perseverance_results['lose_stay_count'] += lose_stay_count
        perseverance_results['total_win_trials'] += win_trials
        perseverance_results['total_lose_trials'] += lose_trials
        perseverance_results['sessions_analyzed'] += 1

        # Calculate session-specific percentages
        win_stay_pct = (win_stay_count / win_trials * 100) if win_trials > 0 else 0
        lose_stay_pct = (lose_stay_count / lose_trials * 100) if lose_trials > 0 else 0
        total_stay_pct = ((win_stay_count + lose_stay_count) / (win_trials + lose_trials) * 100) if (win_trials + lose_trials) > 0 else 0
        
        # Store session data
        session_data.append({
            'session_date': sess,
            'win_stay_count': win_stay_count,
            'win_trials': win_trials,
            'win_stay_pct': win_stay_pct,
            'lose_stay_count': lose_stay_count,
            'lose_trials': lose_trials,
            'lose_stay_pct': lose_stay_pct,
            'total_stay_count': win_stay_count + lose_stay_count,
            'total_trials': win_trials + lose_trials,
            'total_stay_pct': total_stay_pct
        })

        # Print session-specific results
        print(f"  Session {sess} perseverance stats:")
        print(f"    Win-Stay: {win_stay_count}/{win_trials} trials ({win_stay_pct:.1f}%)")
        print(f"    Lose-Stay: {lose_stay_count}/{lose_trials} trials ({lose_stay_pct:.1f}%)")
        print(f"    Overall Stay: {win_stay_count + lose_stay_count}/{win_trials + lose_trials} trials ({total_stay_pct:.1f}%)")
        print()

    # Calculate overall percentages
    if perseverance_results['total_win_trials'] > 0:
        perseverance_results['win_stay_percentage'] = (perseverance_results['win_stay_count'] / perseverance_results['total_win_trials']) * 100
    else:
        perseverance_results['win_stay_percentage'] = 0

    if perseverance_results['total_lose_trials'] > 0:
        perseverance_results['lose_stay_percentage'] = (perseverance_results['lose_stay_count'] / perseverance_results['total_lose_trials']) * 100
    else:
        perseverance_results['lose_stay_percentage'] = 0
        
    total_trials = perseverance_results['total_win_trials'] + perseverance_results['total_lose_trials']
    if total_trials > 0:
        perseverance_results['total_stay_percentage'] = ((perseverance_results['win_stay_count'] + perseverance_results['lose_stay_count']) / 
                                                     total_trials) * 100
    else:
        perseverance_results['total_stay_percentage'] = 0

    # Print overall results
    print("\n=== Perseverance Analysis (Stay Behavior) ===")
    print(f"Subject: {subject_id}")
    print(f"Sessions analyzed: {perseverance_results['sessions_analyzed']}")
    print(f"Total valid trial pairs: {total_trials}")
    print(f"Win-Stay: {perseverance_results['win_stay_count']}/{perseverance_results['total_win_trials']} trials ({perseverance_results['win_stay_percentage']:.1f}%)")
    print(f"Lose-Stay: {perseverance_results['lose_stay_count']}/{perseverance_results['total_lose_trials']} trials ({perseverance_results['lose_stay_percentage']:.1f}%)")
    print(f"Overall Stay: {perseverance_results['win_stay_count'] + perseverance_results['lose_stay_count']}/{total_trials} trials ({perseverance_results['total_stay_percentage']:.1f}%)")

        # Visualization based on whether we're analyzing one or multiple sessions
    if perseverance_results['sessions_analyzed'] > 0:
        plt.figure(figsize=(12, 6))
        
        # If analyzing multiple sessions, plot session-by-session rates
        if len(session_data) > 1:
            # Extract data for plotting
            sessions = list(range(1, perseverance_results['sessions_analyzed'] + 1))
            win_stay_pcts = [s['win_stay_pct'] for s in session_data]
            lose_stay_pcts = [s['lose_stay_pct'] for s in session_data]
            total_stay_pcts = [s['total_stay_pct'] for s in session_data]
            
            # Plot session data
            plt.plot(sessions, win_stay_pcts, 'o-', color='green', label='Win-Stay')
            plt.plot(sessions, lose_stay_pcts, 'o-', color='red', label='Lose-Stay')
            plt.plot(sessions, total_stay_pcts, 'o-', color='blue', linewidth=2, label='Overall Stay')
            
            # Add reference line at 50%
            plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
            
            # Formatting
            plt.xlabel('Session Number')
            plt.ylabel('Stay Percentage (%)')
            plt.title(f'Perseverance Across Sessions {subject_id}')
            plt.xticks(sessions)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            # For single session, create simple line chart with the three values
            categories = ['Win-Stay', 'Lose-Stay', 'Overall']
            stay_percentages = [
                perseverance_results['win_stay_percentage'],
                perseverance_results['lose_stay_percentage'],
                perseverance_results['total_stay_percentage']
            ]
            
            plt.plot(['Win-Stay', 'Lose-Stay', 'Overall'], stay_percentages, 'o-', color='blue', linewidth=2)
            plt.ylabel('Stay Percentage (%)')
            plt.title('Perseverance Analysis')
            plt.ylim(0, 100)
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add data points to the graph
            for i, (pct, count, total) in enumerate(zip(
                stay_percentages[:2],  # Just win-stay and lose-stay
                [perseverance_results['win_stay_count'], perseverance_results['lose_stay_count']],
                [perseverance_results['total_win_trials'], perseverance_results['total_lose_trials']])):
                plt.annotate(f'{count}/{total}\n({pct:.1f}%)', 
                             (i, pct), 
                             textcoords="offset points", 
                             xytext=(0,10), 
                             ha='center')
            
            # Add overall percentage
            plt.annotate(f"{perseverance_results['win_stay_count'] + perseverance_results['lose_stay_count']}/"
                         f"{total_trials}\n({perseverance_results['total_stay_percentage']:.1f}%)",
                         (2, perseverance_results['total_stay_percentage']),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center')
            
        plt.tight_layout()
        
        # Save the figure
        save_figure(plt.gcf(), subject_id, "pooled", "perseverance_analysis")
        plt.show()
    
    perseverance_results['session_data'] = session_data
    return perseverance_results
        
def analyze_group_perseverance(subject_ids=None, behavior_df=None, save_fig=True):
    """
    Calculate and plot average win-stay, lose-stay, and overall stay percentages
    across multiple subjects. Treats each animal as one data point per session.
    
    Parameters:
    -----------
    subject_ids : list, optional
        List of subject IDs to include. If None, uses default list.
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file
    save_fig : bool, optional
        Whether to save the generated figure
        
    Returns:
    --------
    dict: Dictionary with perseverance data across subjects
    """
    # Default subjects if not provided
    if subject_ids is None:
        subject_ids = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
        print(f"Using default subject list: {subject_ids}")
    
    # Store results for each subject
    all_subject_data = {}
    max_sessions = 0
    
    # Process each subject
    for subject_id in subject_ids:
        print(f"Processing {subject_id}...")
        
        # Analyze perseverance for this subject
        result = analyze_win_stay_lose_stay(subject_id, behavior_df=behavior_df)
        
        if result and 'session_data' in result and result['session_data']:
            # Store session data
            all_subject_data[subject_id] = result['session_data']
            
            # Track maximum number of sessions
            max_sessions = max(max_sessions, len(result['session_data']))
    
    if not all_subject_data:
        print("No valid data found for any subjects")
        return None
    
    # Prepare data structure for averaging
    session_averages = {
        'win_stay': [[] for _ in range(max_sessions)],
        'lose_stay': [[] for _ in range(max_sessions)],
        'total_stay': [[] for _ in range(max_sessions)]
    }
    
    # Collect data for each session across subjects
    for subject_id, sessions in all_subject_data.items():
        for i, session_data in enumerate(sessions):
            session_averages['win_stay'][i].append(session_data['win_stay_pct'])
            session_averages['lose_stay'][i].append(session_data['lose_stay_pct'])
            session_averages['total_stay'][i].append(session_data['total_stay_pct'])
    
    # Calculate means and SEMs for each session
    win_stay_means = []
    lose_stay_means = []
    total_stay_means = []
    win_stay_sems = []
    lose_stay_sems = []
    total_stay_sems = []
    subjects_per_session = []
    
    for i in range(max_sessions):
        # Only include sessions with at least 3 subjects' data
        win_data = [x for x in session_averages['win_stay'][i] if not np.isnan(x)]
        lose_data = [x for x in session_averages['lose_stay'][i] if not np.isnan(x)]
        total_data = [x for x in session_averages['total_stay'][i] if not np.isnan(x)]
        
        n_subjects = len(win_data)
        subjects_per_session.append(n_subjects)
        
        # Only calculate stats if we have enough data
        if n_subjects >= 3:
            win_stay_means.append(np.mean(win_data))
            lose_stay_means.append(np.mean(lose_data))
            total_stay_means.append(np.mean(total_data))
            
            win_stay_sems.append(np.std(win_data) / np.sqrt(n_subjects))
            lose_stay_sems.append(np.std(lose_data) / np.sqrt(n_subjects))
            total_stay_sems.append(np.std(total_data) / np.sqrt(n_subjects))
        else:
            # Stop calculating when fewer than 3 subjects have data
            break
    
    # Plot the results
    valid_sessions = len(win_stay_means)
    if valid_sessions > 0:
        plt.figure(figsize=(14, 8))
        
        # Create x values for sessions
        sessions = list(range(1, valid_sessions + 1))
        
        # Plot individual subject data with thinner, semi-transparent lines
        for subject_id, subject_sessions in all_subject_data.items():
            x_vals = range(1, min(len(subject_sessions) + 1, valid_sessions + 1))
            
            # Only plot up to the valid session limit
            win_vals = [s['win_stay_pct'] for s in subject_sessions[:valid_sessions]]
            lose_vals = [s['lose_stay_pct'] for s in subject_sessions[:valid_sessions]]
            total_vals = [s['total_stay_pct'] for s in subject_sessions[:valid_sessions]]
            
            plt.plot(x_vals, win_vals, 'o-', color='green', alpha=0.15, linewidth=0.7)
            plt.plot(x_vals, lose_vals, 'o-', color='red', alpha=0.15, linewidth=0.7)
            plt.plot(x_vals, total_vals, 'o-', color='blue', alpha=0.15, linewidth=0.7)
        
        # Plot the group averages with thicker lines and SEM
        plt.errorbar(sessions, win_stay_means, yerr=win_stay_sems, 
                     color='green', linewidth=2, label='Win-Stay')
        plt.errorbar(sessions, lose_stay_means, yerr=lose_stay_sems, 
                     color='red', linewidth=2, label='Lose-Stay')
        plt.errorbar(sessions, total_stay_means, yerr=total_stay_sems, 
                     color='blue', linewidth=2, label='Overall Stay')
        
        # Add reference line at 50%
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        # Formatting
        plt.xlabel('Session Number', fontsize=14)
        plt.ylabel('Stay Percentage (%)', fontsize=14)
        plt.title(f'Group Perseverance Analysis Across Sessions (n={len(subject_ids)} subjects)', 
                  fontsize=16)
        plt.xticks(sessions)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            save_figure(plt.gcf(), "all_subjects", "group", "perseverance_analysis")
        
        plt.show()
    
    # Return analysis results
    return {
        'subjects': subject_ids,
        'max_sessions': max_sessions,
        'win_stay_means': win_stay_means,
        'lose_stay_means': lose_stay_means, 
        'total_stay_means': total_stay_means,
        'win_stay_sems': win_stay_sems,
        'lose_stay_sems': lose_stay_sems,
        'total_stay_sems': total_stay_sems,
        'subjects_per_session': subjects_per_session,
        'valid_sessions': valid_sessions
    }


def analyze_loss_streaks_before_win(subject_id="All", skipped_missed=True, only_1_5=False, behavior_df=None, specific_subjects=None, plot_trial='loss'):
    """
    Analyze photometry signals for loss streaks of different lengths that end with a win.
    This function identifies trials that were not rewarded but where the next trial was rewarded,
    and categorizes them based on the number of consecutive losses before that trial.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject or "All" for cross-subject analysis
    skipped_missed : bool, optional (default=True)
        If True, filter out missed trials ('M') from streak calculation
        If False, include missed trials as losses as long as reward=0
    only_1_5 : bool, optional (default=False)
        If True, only plot categories 1 and 5+ (shortest and longest streaks)
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    specific_subjects : list, optional
        List of subject IDs to include if subject_id="All"
    plot_trial : str, optional (default='loss')
        Which trial to analyze for each streak:
        - 'win': Plot the first win trial after the loss streak
        - 'loss': Plot the last loss trial in the streak (default)
        
    Returns:
    --------
    dict: Analysis results for different loss streak lengths
    """
    # Handle cross-subject analysis
    if subject_id == "All":
        if specific_subjects is None:
            # Default list of subjects
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Store subject-level data for each streak category
        subject_streak_avgs = {
            '1_loss': [],
            '2_loss': [],
            '3_loss': [],
            '4_loss': [],
            '5plus_loss': []
        }
        
        # Store number of trials in each category per subject
        subject_trial_counts = {
            '1_loss': [],
            '2_loss': [],
            '3_loss': [],
            '4_loss': [],
            '5plus_loss': []
        }
        
        time_axis = None
        total_trials = 0
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for loss streaks before win analysis...")
            
            # Process individual subject
            subj_result = analyze_loss_streaks_before_win_single(subj, skipped_missed, only_1_5, behavior_df, plot_trial)
            
            if subj_result and 'streak_averages' in subj_result:
                if time_axis is None:
                    time_axis = subj_result['time_axis']
                
                total_trials += subj_result['total_trials']
                
                # Add each streak category's average to the subject-level collection
                for category in ['1_loss', '2_loss', '3_loss', '4_loss', '5plus_loss']:
                    if category in subj_result['streak_averages']:
                        subject_streak_avgs[category].append(subj_result['streak_averages'][category])
                        subject_trial_counts[category].append(len(subj_result['streak_data'][category]))
                    else:
                        # If this category doesn't exist for this subject, add empty placeholder
                        subject_trial_counts[category].append(0)
        
        # Check if we have enough data to create a plot
        if time_axis is None or not any(subject_streak_avgs.values()):
            print("No valid loss streak data found for cross-subject analysis")
            return None
            
        # Create the plot
        plt.figure(figsize=(12, 7))
        
        # Define colors and labels
        colors = {
            '1_loss': 'blue',
            '2_loss': 'green',
            '3_loss': 'orange',
            '4_loss': 'red',
            '5plus_loss': 'purple'
        }
        
        # Determine which categories to plot based on only_1_5 parameter
        categories_to_plot = ['1_loss', '5plus_loss'] if only_1_5 else ['1_loss', '2_loss', '3_loss', '4_loss', '5plus_loss']
        
        # Calculate and plot the cross-subject average for each streak category
        displayed_trials = 0
        
        for category in categories_to_plot:
            if len(subject_streak_avgs[category]) > 0:
                # Calculate mean and SEM across subjects that have this streak category
                category_mean = np.mean(subject_streak_avgs[category], axis=0)
                category_sem = np.std(subject_streak_avgs[category], axis=0) / np.sqrt(len(subject_streak_avgs[category]))
                
                # Calculate total trials for this category across subjects
                category_trial_count = sum(subject_trial_counts[category])
                displayed_trials += category_trial_count
                
                # Format label based on the category and plot_trial
                if plot_trial == 'win':
                    # For win trials, label shows this is the win following a loss streak
                    if category == '1_loss':
                        label_name = f"Win after 1 Loss (n={category_trial_count}, {len(subject_streak_avgs[category])} subjects)"
                    elif category == '5plus_loss':
                        label_name = f"Win after 5+ Consecutive Losses (n={category_trial_count}, {len(subject_streak_avgs[category])} subjects)"
                    else:
                        label_name = f"Win after {category[0]} Consecutive Losses (n={category_trial_count}, {len(subject_streak_avgs[category])} subjects)"
                else:
                    # For loss trials, label shows this is the last loss in the streak
                    if category == '1_loss':
                        label_name = f"1 Loss (n={category_trial_count}, {len(subject_streak_avgs[category])} subjects)"
                    elif category == '5plus_loss':
                        label_name = f"Last of 5+ Consecutive Losses (n={category_trial_count}, {len(subject_streak_avgs[category])} subjects)"
                    else:
                        label_name = f"Last of {category[0]} Consecutive Losses (n={category_trial_count}, {len(subject_streak_avgs[category])} subjects)"
                
                plt.fill_between(time_axis,
                               category_mean - category_sem,
                               category_mean + category_sem,
                               color=colors[category], alpha=0.3)
                plt.plot(time_axis, category_mean,
                       color=colors[category], linewidth=2,
                       label=label_name)
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Labels and formatting
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('ΔF/F', fontsize=16)
        
        # Add appropriate title based on parameters
        missed_text = "excluding" if skipped_missed else "including"
        plot_type = "Short (1) vs. Long (5+)" if only_1_5 else "All Categories"
        
        if plot_trial == 'win':
            plt.title(f'Cross-Subject LC Signal Following Loss Streaks: {plot_type}',
                      fontsize=20)
        else:
            plt.title(f'Cross-Subject LC Signal During Loss Streaks: {plot_type}',
                      fontsize=20)
                  
        plt.xlim([-pre_cue_time, post_cue_time])
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        # Add total trials information
        plt.figtext(0.02, 0.02, f"Total trials analyzed: {displayed_trials} (of {total_trials})", fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the figure
        plot_cat_text = "1_and_5" if only_1_5 else "all_cats"
        trial_suffix = "win_after" if plot_trial == 'win' else "last_loss"
        save_figure(plt.gcf(), "all_subjects", "pooled", f"loss_streaks_{trial_suffix}_{missed_text}_missed_{plot_cat_text}")
        
        plt.show()
        
        # Return analysis results
        return {
            'subject_id': 'All',
            'specific_subjects': specific_subjects,
            'subject_streak_avgs': subject_streak_avgs,
            'subject_trial_counts': subject_trial_counts,
            'time_axis': time_axis,
            'total_trials': total_trials,
            'displayed_trials': displayed_trials,
            'skipped_missed': skipped_missed,
            'only_1_5': only_1_5,
            'plot_trial': plot_trial
        }
        
    else:
        # Original single-subject analysis
        return analyze_loss_streaks_before_win_single(subject_id, skipped_missed, only_1_5, behavior_df, plot_trial)


def analyze_loss_streaks_before_win_single(subject_id, skipped_missed=True, only_1_5=False, behavior_df=None, plot_trial='loss'):
    """
    Analyze photometry signals for loss streaks of different lengths that end with a win.
    This function identifies trials that were not rewarded but where the next trial was rewarded,
    and categorizes them based on the number of consecutive losses before that trial.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    skipped_missed : bool, optional (default=True)
        If True, filter out missed trials ('M') from streak calculation
        If False, include missed trials as losses as long as reward=0
    only_1_5 : bool, optional (default=False)
        If True, only plot categories 1 and 5+ (shortest and longest streaks)
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    plot_trial : str, optional (default='loss')
        Which trial to analyze for each streak:
        - 'win': Plot the first win trial after the loss streak
        - 'loss': Plot the last loss trial in the streak (default)
        
    Returns:
    --------
    dict: Analysis results for different loss streak lengths
    """
    # Find all session directories for this subject
    subject_path = os.path.join(base_dir, subject_id)
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                    if os.path.isdir(os.path.join(subject_path, d)) and
                  os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                  d in matching_pennies_sessions])

    # Store data for each loss streak category
    streak_data = {
        '1_loss': [],  # T0 loss, T-1 no loss
        '2_loss': [],  # T0 & T-1 loss, T-2 no loss
        '3_loss': [],  # T0, T-1, T-2 loss, T-3 no loss
        '4_loss': [],  # T0, T-1, T-2, T-3 loss, T-4 no loss
        '5plus_loss': []  # T0 loss preceded by 4+ losses
    }

    time_axis = None  # Will be set from the first valid session

    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue

        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])

        # Get the full trial data - including photometry for all valid trials
        # (both missed and non-missed that have good photometry recordings)
        all_valid_trials = session_result['valid_trials']
        all_valid_epoched_data = session_result['epoched_data'][all_valid_trials]

        # Filter behavioral data based on skipped_missed parameter
        if skipped_missed:
            # Create a mask where choice is not 'M'
            valid_mask = choices != 'M'
            # Apply filter to raw behavior data
            filtered_rewards = rewards[valid_mask]
            filtered_indices = np.where(valid_mask)[0]
            # Create a mapping from filtered indices to original indices
            filtered_to_orig = {i: filtered_indices[i] for i in range(len(filtered_indices))}
            # Create reverse mapping from original to filtered indices
            orig_to_filtered = {filtered_indices[i]: i for i in range(len(filtered_indices))}
        else:
            # Use all trials without filtering
            filtered_rewards = rewards
            filtered_to_orig = {i: i for i in range(len(rewards))}
            orig_to_filtered = {i: i for i in range(len(rewards))}

        # Skip if session has too few trials
        if len(filtered_rewards) < 6:  # Need at least 6 trials to determine 5+ loss streak
            print(f"Skipping {subject_id}/{session_date}, insufficient trials after filtering")
            continue

        # Find trials that were losses followed by a win in the filtered behavioral data
        for i in range(len(filtered_rewards) - 1):
            # Check if current trial is a loss and next is a win
            if filtered_rewards[i] == 0 and filtered_rewards[i + 1] == 1:
                # This is a loss trial followed by a win
                orig_loss_idx = filtered_to_orig[i]  # Last loss trial
                orig_win_idx = filtered_to_orig[i + 1]  # First win trial after streak

                # Determine which trial to analyze based on plot_trial parameter
                orig_trial_idx = orig_win_idx if plot_trial == 'win' else orig_loss_idx

                # Skip if we don't have photometry data for this trial
                if orig_trial_idx not in all_valid_trials:
                    continue

                # Get the photometry data for this trial
                # Find the index in the valid_trials array
                valid_trial_idx = np.where(np.array(all_valid_trials) == orig_trial_idx)[0]
                if len(valid_trial_idx) == 0:
                    # No photometry data for this trial
                    continue

                photometry_data = all_valid_epoched_data[valid_trial_idx[0]]

                # Now count consecutive losses going backward from the loss trial
                loss_streak = 1  # Start with 1 (current trial is a loss)

                if skipped_missed:
                    # Looking back in filtered space (no missed trials)
                    for j in range(i - 1, -1, -1):
                        if filtered_rewards[j] == 0:
                            loss_streak += 1
                        else:
                            # Found a win, streak ends
                            break
                else:
                    # Looking back in original space (can include missed trials)
                    for j in range(orig_loss_idx - 1, -1, -1):
                        if j < len(rewards) and rewards[j] == 0:
                            loss_streak += 1
                        else:
                            # Found a win, streak ends
                            break

                # Categorize the trial based on streak length
                if loss_streak == 1:
                    streak_data['1_loss'].append(photometry_data)
                elif loss_streak == 2:
                    streak_data['2_loss'].append(photometry_data)
                elif loss_streak == 3:
                    streak_data['3_loss'].append(photometry_data)
                elif loss_streak == 4:
                    streak_data['4_loss'].append(photometry_data)
                else:  # 5 or more consecutive losses
                    streak_data['5plus_loss'].append(photometry_data)

    # Check if we found any valid streaks
    total_trials = sum(len(data) for data in streak_data.values())
    if total_trials == 0:
        print(f"No valid loss streaks found for {subject_id}")
        return None

    # Convert lists to numpy arrays for each category (if not empty)
    for category in streak_data:
        if streak_data[category]:
            streak_data[category] = np.array(streak_data[category])

    # Calculate averages and SEM for each streak length
    streak_averages = {}
    streak_sems = {}
    for category, data in streak_data.items():
        if len(data) > 0:
            streak_averages[category] = np.mean(data, axis=0)
            streak_sems[category] = calculate_sem(data, axis=0)

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Define colors and labels with trial counts
    colors = {
        '1_loss': 'blue',
        '2_loss': 'green',
        '3_loss': 'orange',
        '4_loss': 'red',
        '5plus_loss': 'purple'
    }

    # Create labels based on plot_trial parameter
    if plot_trial == 'win':
        labels = {
            '1_loss': f"Win after 1 Loss (n={len(streak_data['1_loss'])})",
            '2_loss': f"Win after 2 Consecutive Losses (n={len(streak_data['2_loss'])})",
            '3_loss': f"Win after 3 Consecutive Losses (n={len(streak_data['3_loss'])})",
            '4_loss': f"Win after 4 Consecutive Losses (n={len(streak_data['4_loss'])})",
            '5plus_loss': f"Win after 5+ Consecutive Losses (n={len(streak_data['5plus_loss'])})"
        }
    else:
        labels = {
            '1_loss': f"1 Loss (n={len(streak_data['1_loss'])})",
            '2_loss': f"2 Consecutive Losses (n={len(streak_data['2_loss'])})",
            '3_loss': f"3 Consecutive Losses (n={len(streak_data['3_loss'])})",
            '4_loss': f"4 Consecutive Losses (n={len(streak_data['4_loss'])})",
            '5plus_loss': f"5+ Consecutive Losses (n={len(streak_data['5plus_loss'])})"
        }

    # Determine which categories to plot based on only_1_5 parameter
    categories_to_plot = ['1_loss', '5plus_loss'] if only_1_5 else ['1_loss', '2_loss', '3_loss', '4_loss', '5plus_loss']

    # Plot selected streak categories
    for category in categories_to_plot:
        if category in streak_averages and len(streak_data[category]) > 0:
            plt.fill_between(time_axis,
                             streak_averages[category] - streak_sems[category],
                             streak_averages[category] + streak_sems[category],
                             color=colors[category], alpha=0.3)
            plt.plot(time_axis, streak_averages[category],
                     color=colors[category], linewidth=2, label=labels[category])

    # Add vertical line at cue onset
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels and formatting
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('ΔF/F', fontsize=16)

    # Title based on plot_trial parameter
    missed_text = "excluding" if skipped_missed else "including"
    plot_cat_text = "1_and_5" if only_1_5 else "all_cats"
    
    if plot_trial == 'win':
        plt.title(f'LC Signal Following Loss Streaks: {subject_id}',
                 fontsize=20)
    else:
        plt.title(f'LC Signal for Cumulative Loss: {subject_id}',
                 fontsize=20)
                 
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Add total trials information
    displayed_trials = sum(len(streak_data[cat]) for cat in categories_to_plot)
    plt.figtext(0.02, 0.02, f"Total trials analyzed: {displayed_trials} (of {total_trials})", fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

    # Save the figure
    trial_suffix = "win_after" if plot_trial == 'win' else "last_loss"
    save_figure(plt.gcf(), subject_id, "pooled", f"loss_streaks_{trial_suffix}_{missed_text}_missed_{plot_cat_text}")

    plt.show()

    # Return analysis results
    return {
        'subject_id': subject_id,
        'streak_data': streak_data,
        'streak_averages': streak_averages,
        'streak_sems': streak_sems,
        'time_axis': time_axis,
        'total_trials': total_trials,
        'displayed_trials': displayed_trials,
        'skipped_missed': skipped_missed,
        'only_1_5': only_1_5,
        'plot_trial': plot_trial
    }

def analyze_session_win_loss_difference_heatmap(subject_id, comp_conf=False, behavior_df=None):
    """
    Create a heatmap visualization of win-loss signal differences across sessions

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    comp_conf : bool, optional (default=False)
        If True, sorts sessions by computer confidence rather than chronologically

    Returns:
    --------
    dict: Analysis results including win-loss differences across all sessions
    """
    # Get all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, we assume it's already filtered for MatchingPennies protocol
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file with protocol filtering
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                    if os.path.isdir(os.path.join(subject_path, d)) and
                    os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                    d in matching_pennies_sessions])

    # Store results for each session
    session_differences = []
    session_dates = []
    time_axis = None
    peak_differences = []
    session_confidences = {}

    # Load parquet data for confidence calculations if needed
    if comp_conf and behavior_df is None:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            df = df[df['protocol'].str.contains('MatchingPennies', na=False)]
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting
    else:
        df = behavior_df  

    # Process each session
    for session_date in sessions:
        # Process the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df, z_score=False)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue

        if len(session_result['non_m_trials']) < 100:
            print(
                f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        # Filter out missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])

        # Get reward outcomes and photometry data
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        session_plots = session_result['plotting_data']

        # Separate rewarded and unrewarded trials
        rewarded_trials = session_plots[reward_outcomes == 1]
        unrewarded_trials = session_plots[reward_outcomes == 0]

        if len(rewarded_trials) == 0 or len(unrewarded_trials) == 0:
            print(f"Skipping {subject_id}/{session_date}, missing reward outcomes.")
            continue

        # Compute average rewarded and unrewarded signals
        rewarded_avg = np.mean(rewarded_trials, axis=0)
        unrewarded_avg = np.mean(unrewarded_trials, axis=0)

        rewarded_avg = rewarded_avg + np.abs(np.min([rewarded_avg, unrewarded_avg]))
        unrewarded_avg = unrewarded_avg + np.abs(np.min([rewarded_avg, unrewarded_avg]))

        # Compute win-loss difference
        win_loss_diff = rewarded_avg - unrewarded_avg

        # Store session information
        session_differences.append(win_loss_diff)
        session_dates.append(session_date)

        # Store time axis (same for all sessions)
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Calculate peak difference (maximum absolute difference in post-cue window)
        post_cue_indices = time_axis > 0
        peak_diff = np.max(np.abs(win_loss_diff[post_cue_indices]))
        peak_differences.append(peak_diff)
        
        # Calculate computer confidence if requested
        if comp_conf:
            try:
                # Get data for this session
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
                
                if not session_df.empty and 'min_pvalue' in session_df.columns:
                    # Extract p-values
                    p_values = session_df['min_pvalue'].values
                    
                    # Remove NaN values
                    p_values = p_values[~np.isnan(p_values)]
                    
                    if len(p_values) > 0:
                        # Cap very small p-values at 10^-12 to avoid infinite confidence
                        min_p_value = 1e-12
                        p_values = np.maximum(p_values, min_p_value)
                        
                        # Calculate confidence as -log10(p_value)
                        confidence = -np.log10(p_values)
                        
                        # Calculate average confidence for the session
                        avg_confidence = np.mean(confidence)
                        session_confidences[session_date] = avg_confidence
                        print(f"Session {session_date} average confidence: {avg_confidence:.4f}")
                else:
                    print(f"No min_pvalue data found for {subject_id}/{session_date}")
                    
            except Exception as e:
                print(f"Error calculating confidence for {subject_id}/{session_date}: {e}")

    if not session_differences:
        print(f"No valid sessions found for {subject_id}")
        return None
    chronological_indices = list(range(len(session_dates)))
    chronological_order = {date: idx+1 for idx, date in enumerate(session_dates)}
    
    # Sort sessions based on confidence or chronologically
    if comp_conf and session_confidences:
        # Filter out sessions without confidence values
        valid_sessions = [s for s in session_dates if s in session_confidences]
        
        if len(valid_sessions) < len(session_dates):
            print(f"Warning: Only {len(valid_sessions)} of {len(session_dates)} sessions have confidence values")
        
        if not valid_sessions:
            print("No sessions with valid confidence values found, using chronological sorting")
            sorted_indices = chronological_indices
        else:
            # Create mapping from session date to index
            date_to_idx = {date: idx for idx, date in enumerate(session_dates)}
            
            # Sort sessions by confidence (highest first)
            sorted_session_dates = sorted(valid_sessions, key=lambda s: session_confidences.get(s, 0), reverse=True)
            
            # Get indices in the original data
            sorted_indices = [date_to_idx[date] for date in sorted_session_dates]
            
            # Print the confidence ranking
            print("\nSessions ranked by computer confidence (highest to lowest):")
            for i, sess in enumerate(sorted_session_dates):
                # Use chronological order (original session number) in the printout
                print(f"{i+1}. Session {chronological_order[sess]} (date: {sess}): {session_confidences[sess]:.4f}")
    else:
        # Use chronological sorting (default)
        sorted_indices = chronological_indices

    # Reorder arrays based on sorted indices
    session_differences = [session_differences[i] for i in sorted_indices]
    session_dates = [session_dates[i] for i in sorted_indices]
    peak_differences = [peak_differences[i] for i in sorted_indices]

    # Convert to array for heatmap
    win_loss_array = np.array(session_differences)

    # Create figure for heatmap
    fig = plt.figure(figsize=(18, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])

    # Plot heatmap of win-loss differences (top)
    ax_heatmap = fig.add_subplot(gs[0])

    # Flip the array vertically for display (earliest/lowest confidence at bottom)
    win_loss_array = np.flipud(win_loss_array)
    flipped_session_dates = session_dates[::-1]  # Reverse the session dates for y-axis labels

    # Special coloring for JOA-M-0020
    if subject_id == "JOA-M-0020":
        # Find the minimum value (should be slightly below 0)
        min_val = np.min(win_loss_array)
        max_val = np.max(win_loss_array)
        
        # Create a custom white-to-red colormap with better differentiation
        from matplotlib.colors import LinearSegmentedColormap
        
        # Define a custom colormap with more distinct reds
        colors = [(1, 1, 1),          # White
                (1, 0.8, 0.8),      # Very light red
                (1, 0.6, 0.6),      # Light red
                (1, 0.3, 0.3),      # Medium red
                (0.8, 0, 0),        # Dark red
                (0.5, 0, 0)]        # Very dark red/maroon
        
        cmap_name = 'WhiteToRedEnhanced'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
        
        # Create the heatmap with the custom white-to-red gradient
        # White for minimum/zero values, dark red for maximum positive values
        im = ax_heatmap.imshow(win_loss_array,
                            aspect='auto',
                            extent=[time_axis[0], time_axis[-1], 0, len(session_differences)],
                            origin='lower',
                            cmap=custom_cmap,  # Custom white-to-red gradient
                            interpolation='nearest',
                            vmin=min_val,      # Set minimum value as white
                        vmax=max_val)  
    else:
        # Regular symmetric color scaling around zero for other subjects
        max_abs_val = np.max(np.abs(win_loss_array))
        
        im = ax_heatmap.imshow(win_loss_array,
                            aspect='auto',
                            extent=[time_axis[0], time_axis[-1], 0, len(session_differences)],
                            origin='lower',
                            cmap='RdBu_r',
                            interpolation='nearest',
                            vmin=-max_abs_val,
                            vmax=max_abs_val)

    # Add vertical line at cue onset
    ax_heatmap.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    # Labels and formatting
    sort_type = "Computer Confidence" if comp_conf else "Chronological"
    ax_heatmap.set_xlabel('Time (s) from first lick', fontsize=24)
    ax_heatmap.set_ylabel('Session', fontsize=24)
    ax_heatmap.set_title(f'Win-Loss Signal Difference Across Sessions: {subject_id} ({sort_type})', fontsize=26)
    ax_heatmap.tick_params(axis='both', which='major', labelsize=20)

    # Add specific y-tick labels at regular intervals
    tick_step = max(1, len(session_dates) // 10)  # Show at most 10 session labels
    y_ticks = np.arange(0.5, len(session_dates), tick_step)

    # Create labels for every tick_step interval
    y_label_with_num = []
    for i in range(0, len(flipped_session_dates), tick_step):
        session_date = flipped_session_dates[i]
        chrono_num = chronological_order[session_date]  # Original chronological number
        y_label_with_num.append(f"{chrono_num}")

    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(y_label_with_num)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    if subject_id == "JOA-M-0020":
        cbar.set_label('Win-Loss ΔF/F Difference', fontsize=24)
    else:
        cbar.set_label('Win-Loss ΔF/F Difference', fontsize=24)

    # Save figure
    sort_suffix = "by_comp_conf" if comp_conf else "chronological"
    save_figure(fig, subject_id, "all_sessions", f"win_loss_difference_heatmap_{sort_suffix}")

    plt.show()

    # Return analysis results
    return {
        'subject_id': subject_id,
        'session_dates': session_dates,
        'time_axis': time_axis,
        'win_loss_differences': session_differences,
        'peak_differences': peak_differences,
        'session_confidences': session_confidences if comp_conf else None
    }


def analyze_session_average_heatmap(subject_id, comp_conf=False, behavior_df=None):
    """
    Create a heatmap visualization of average photometry signals across sessions

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    comp_conf : bool, optional (default=False)
        If True, sorts sessions by computer confidence rather than chronologically

    Returns:
    --------
    dict: Analysis results including average signals across all sessions
    """
    # Get all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)

    # Get matching pennies sessions from behavior dataframe if provided, otherwise load from parquet
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                    if os.path.isdir(os.path.join(subject_path, d)) and
                    os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                    d in matching_pennies_sessions])

    # Store results for each session
    session_averages = []
    session_dates = []
    time_axis = None
    peak_averages = []
    session_confidences = {}

    # Load parquet data for confidence calculations if needed
    if comp_conf and behavior_df is None:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            df = df[df['protocol'].str.contains('MatchingPennies', na=False)]
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting
    else:
        df = behavior_df  

    # Process each session
    for session_date in sessions:
        # Process the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue

        if len(session_result['non_m_trials']) < 100:
            print(
                f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        # Get photometry data from all non-missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])
        session_plots = session_result['plotting_data']

        if len(session_plots) == 0:
            print(f"Skipping {subject_id}/{session_date}, no photometry data.")
            continue

        # Compute average signal across all trials
        session_avg = np.mean(session_plots, axis=0)

        # Store session information
        session_averages.append(session_avg)
        session_dates.append(session_date)

        # Store time axis (same for all sessions)
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Calculate peak average (maximum signal in post-cue window)
        post_cue_indices = time_axis > 0
        peak_avg = np.max(session_avg[post_cue_indices])
        peak_averages.append(peak_avg)
        
        # Calculate computer confidence if requested
        if comp_conf:
            try:
                # Get data for this session
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
                
                if not session_df.empty and 'min_pvalue' in session_df.columns:
                    # Extract p-values
                    p_values = session_df['min_pvalue'].values
                    
                    # Remove NaN values
                    p_values = p_values[~np.isnan(p_values)]
                    
                    if len(p_values) > 0:
                        # Cap very small p-values at 10^-12 to avoid infinite confidence
                        min_p_value = 1e-12
                        p_values = np.maximum(p_values, min_p_value)
                        
                        # Calculate confidence as -log10(p_value)
                        confidence = -np.log10(p_values)
                        
                        # Calculate average confidence for the session
                        avg_confidence = np.mean(confidence)
                        session_confidences[session_date] = avg_confidence
                        print(f"Session {session_date} average confidence: {avg_confidence:.4f}")
                else:
                    print(f"No min_pvalue data found for {subject_id}/{session_date}")
                    
            except Exception as e:
                print(f"Error calculating confidence for {subject_id}/{session_date}: {e}")

    if not session_averages:
        print(f"No valid sessions found for {subject_id}")
        return None

    # Sort sessions based on confidence or chronologically
    if comp_conf and session_confidences:
        # Filter out sessions without confidence values
        valid_sessions = [s for s in session_dates if s in session_confidences]
        
        if len(valid_sessions) < len(session_dates):
            print(f"Warning: Only {len(valid_sessions)} of {len(session_dates)} sessions have confidence values")
        
        if not valid_sessions:
            print("No sessions with valid confidence values found, using chronological sorting")
            sorted_indices = list(range(len(session_dates)))
        else:
            # Create mapping from session date to index
            date_to_idx = {date: idx for idx, date in enumerate(session_dates)}
            
            # Sort sessions by confidence (highest first)
            sorted_session_dates = sorted(valid_sessions, key=lambda s: session_confidences.get(s, 0), reverse=True)
            
            # Get indices in the original data
            sorted_indices = [date_to_idx[date] for date in sorted_session_dates]
            
            # Print the confidence ranking
            print("\nSessions ranked by computer confidence (highest to lowest):")
            for i, sess in enumerate(sorted_session_dates):
                print(f"{i+1}. Session {sess}: {session_confidences[sess]:.4f}")
    else:
        # Use chronological sorting (default)
        sorted_indices = list(range(len(session_dates)))

    # Reorder arrays based on sorted indices
    session_averages = [session_averages[i] for i in sorted_indices]
    session_dates = [session_dates[i] for i in sorted_indices]
    peak_averages = [peak_averages[i] for i in sorted_indices]

    # Convert to array for heatmap
    avg_signal_array = np.array(session_averages)

    # Create figure for heatmap and peak averages plot
    fig = plt.figure(figsize=(18, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])

    # Plot heatmap of average signals (top)
    ax_heatmap = fig.add_subplot(gs[0])

    # Flip the array vertically for display (earliest/lowest confidence at bottom)
    avg_signal_array = np.flipud(avg_signal_array)
    flipped_session_dates = session_dates[::-1]  # Reverse the session dates for y-axis labels

    # Create the heatmap with blue-to-yellow-to-white colormap
    im = ax_heatmap.imshow(avg_signal_array,
                           aspect='auto',
                           extent=[time_axis[0], time_axis[-1], 0, len(session_averages)],
                           origin='lower',
                           cmap='viridis',  # blue-low, yellow/white-high
                           interpolation='nearest')

    # Add vertical line at cue onset
    ax_heatmap.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    # Labels and formatting
    sort_type = "Computer Confidence" if comp_conf else "Chronological (oldest first)"
    ax_heatmap.set_xlabel('Time (s)', fontsize=12)
    ax_heatmap.set_ylabel('Session', fontsize=12)
    ax_heatmap.set_title(f'Average LC Signal Across Sessions: {subject_id} (sorted by {sort_type})', fontsize=20)

    # Add specific y-tick labels at regular intervals
    tick_step = max(1, len(session_dates) // 10)  # Show at most 10 session labels
    y_ticks = np.arange(0.5, len(session_dates), tick_step)

    # Create labels for every tick_step interval
    y_label_with_num = []
    for i in range(0, len(flipped_session_dates), tick_step):
        if comp_conf and flipped_session_dates[i] in session_confidences:
            y_label_with_num.append(f"{i+1}: {flipped_session_dates[i]} (conf: {session_confidences[flipped_session_dates[i]]:.2f})")
        else:
            y_label_with_num.append(f"{i+1}: {flipped_session_dates[i]}")

    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(y_label_with_num)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label('ΔF/F', fontsize=10)


    # Save figure
    sort_suffix = "by_comp_conf" if comp_conf else "chronological"
    save_figure(fig, subject_id, "all_sessions", f"average_signal_heatmap_{sort_suffix}")

    plt.show()

def add_statistical_test(subject_id, quartile_labels, all_choice_switches, quartile_trial_counts,
                         quartile_switch_rates):
    """
    Add statistical analysis to the loss trials signal quartiles analysis.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    quartile_labels : numpy.ndarray
        Array containing quartile labels (0-3) for each trial
    all_choice_switches : numpy.ndarray
        Boolean array indicating whether a switch occurred for each trial
    quartile_trial_counts : list
        List of trial counts for each quartile
    quartile_switch_rates : list
        List of switch percentages for each quartile
    """

    print("\n=== Statistical Analysis of Switch Rates Across Quartiles ===")

    contingency_table = []

    for quartile in range(4):
        quartile_mask = quartile_labels == quartile
        trials_in_quartile = np.sum(quartile_mask)

        if trials_in_quartile > 0:
            switch_count = np.sum(all_choice_switches[quartile_mask])
            stay_count = trials_in_quartile - switch_count
            contingency_table.append([switch_count, stay_count])
        else:
            contingency_table.append([0, 0])

    contingency_table = np.array(contingency_table)

    # Chi-square test for independence
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-squared test for independence:")
        print(f"  Chi2 = {chi2:.3f}, p = {p_chi2:.6f}, dof = {dof}")

        if p_chi2 < 0.05:
            print("  Significant difference in switch rates across quartiles (p < 0.05)")
        else:
            print("  No significant difference in switch rates across quartiles")

        # Check if any expected frequencies are too low for reliable chi-square
        if np.any(expected < 5):
            print("  Warning: Some expected frequencies are < 5, chi-square may not be reliable")
            print("  Consider using Fisher's exact test for pairwise comparisons")
    except Exception as e:
        print(f"  Error in chi-squared test: {e}")

    # Pairwise comparisons between highest and lowest quartiles
    try:
        # Compare quartile 1 (lowest signal) vs quartile 4 (highest signal)
        q1_data = contingency_table[0]
        q4_data = contingency_table[3]

        # Reshape to 2x2 for Fisher's exact test
        fisher_table = np.array([q1_data, q4_data])

        odds_ratio, p_fisher = fisher_exact(fisher_table)
        print(f"\nFisher's exact test: Quartile 1 vs Quartile 4")
        print(f"  Odds ratio = {odds_ratio:.3f}, p = {p_fisher:.6f}")

        if p_fisher < 0.05:
            print("  Significant difference between quartile 1 and quartile 4 (p < 0.05)")
            # Report which quartile has higher switch rate
            if quartile_switch_rates[0] > quartile_switch_rates[3]:
                print(
                    f"  Quartile 1 has a higher switch rate ({quartile_switch_rates[0]:.1f}%) than Quartile 4 ({quartile_switch_rates[3]:.1f}%)")
            else:
                print(
                    f"  Quartile 4 has a higher switch rate ({quartile_switch_rates[3]:.1f}%) than Quartile 1 ({quartile_switch_rates[0]:.1f}%)")
        else:
            print("  No significant difference between quartile 1 and quartile 4")
    except Exception as e:
        print(f"  Error in Fisher's exact test: {e}")

    print("\nContingency Table (Switch vs Stay for each quartile):")
    print("             Switches  Stays   Total   Switch%")
    for i, (switch, stay) in enumerate(contingency_table):
        total = switch + stay
        percent = (switch / total * 100) if total > 0 else 0
        print(f"Quartile {i + 1}:  {switch:6d}   {stay:6d}   {total:5d}   {percent:.1f}%")

def analyze_signal_quartiles_by_outcome(subject_id, signal_window='pre_cue', condition='loss', plot_verification=True, behavior_df=None):
    """
    Analyze trials based on photometry signal in specified time window and determine choice switching behavior.
    Works with both win and loss trials as the condition.
    
    For post-cue windows ('early_post' and 'late_post'):
    - Selects current win/loss trials (T0) and sorts them by signal in the time window
    - Calculates % of trials where the next choice (T+1) is different from current choice (T0)
    
    For pre-cue window ('pre_cue'):
    - Selects trials that follow a win/loss (T0 follows T-1 win/loss) and sorts by pre-cue signal
    - Calculates % of trials where the current choice (T0) is different from previous choice (T-1)
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    signal_window : str, optional (default='pre_cue')
        Time window to use for calculating average photometry signal:
        - 'pre_cue': -0.75s to -0.25s before lick (-0.75 to -0.25)
        - 'early_post': +1s to +2s after lick (1 to 2)
        - 'late_post': +3.5s to +4.5s after lick (3.5 to 4.5)
    condition : str, optional (default='loss')
        Which trial outcome to analyze:
        - 'loss': Analyze trials with no reward (0)
        - 'win': Analyze trials with reward (1)
    plot_verification : bool, optional (default=True)
        Whether to create verification plots showing the sorted quartiles
        
    Returns:
    --------
    dict: Analysis results including quartile switch rates and plotted data
    """
    # Define time windows based on the parameter
    time_windows = {
        'pre_cue': (-0.75, -0.25),
        'early_post': (1.0, 2.0),
        'late_post': (3.5, 4.5)
    }
    
    if signal_window not in time_windows:
        raise ValueError(f"Invalid signal_window. Choose from: {list(time_windows.keys())}")
        
    if condition not in ['win', 'loss']:
        raise ValueError("Condition must be either 'win' or 'loss'")
        
    # Get time window bounds
    window_start, window_end = time_windows[signal_window]
    
    # Determine analysis mode based on time window
    is_pre_cue_analysis = signal_window == 'pre_cue'
    
    # Store trial data and corresponding behavior
    all_trials_data = []         # Photometry data for all selected trials
    all_trial_signals = []       # Average signal in window for selected trials
    all_choice_switches = []     # Boolean: True if switch in choice
    all_next_reward = []         # Boolean: True if next trial rewarded 
    all_state_probs = []         # Store state probabilities for each trial
    time_axis = None             # Will be set from the first valid session
    
    # Set the reward value we're looking for based on condition
    target_reward = 1 if condition == 'win' else 0
    
    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None

    # Get matching pennies sessions from behavior dataframe if provided, otherwise load from parquet
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")
    
    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_dir)
                if os.path.isdir(os.path.join(subject_dir, d)) and
                os.path.exists(os.path.join(subject_dir, d, "deltaff.npy")) and
                d in matching_pennies_sessions])
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue
            
        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']
            
        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])
        
        # Skip sessions with too few trials
        min_required = 3  # Need at least 3 trials for a meaningful analysis
        if len(rewards) < min_required:
            print(f"Skipping {subject_id}/{session_date}, insufficient trials")
            continue
            
        # Filter out missed trials
        non_miss_mask = choices != 'M'
        non_miss_rewards = rewards[non_miss_mask]
        non_miss_choices = choices[non_miss_mask]
        
        # Create mapping from filtered indices to original indices
        non_miss_indices = np.where(non_miss_mask)[0]
        filtered_to_orig = {i: non_miss_indices[i] for i in range(len(non_miss_indices))}
        
        # Get photometry data for valid trials
        valid_trials = session_result['valid_trials']
        epoched_data = session_result['epoched_data'][valid_trials]
        
        # Find time indices for the specified window
        window_idx_start = np.where(time_axis >= window_start)[0][0]
        window_idx_end = np.where(time_axis <= window_end)[0][-1]
        
        # Get state probabilities for this session if parquet data is available
        if behavior_df is not None:
            # Filter from the provided dataframe (which is already filtered by protocol)
            session_df = behavior_df[(behavior_df['subjid'] == subject_id) & 
                                    (behavior_df['date'] == session_date)]
        else:
            # This is a fallback if no behavior_df is provided
            try:
                # Load from parquet file and filter
                df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                df['date'] = df['date'].astype(str)
                session_df = df[(df['subjid'] == subject_id) & 
                            (df['date'] == session_date) & 
                            (df["ignore"] == 0) & 
                            (df['protocol'].str.contains('MatchingPennies', na=False))]
            except Exception as e:
                print(f"Error loading parquet data: {e}")
                session_df = None
        
        if is_pre_cue_analysis:
            # PRE-CUE ANALYSIS MODE
            # Look at trials AFTER a win/loss and analyze if the choice switched from the previous trial
            
            for i in range(1, len(non_miss_rewards)):  # Start from second trial
                prev_trial_idx = i - 1
                curr_trial_idx = i
                
                # Check if previous trial matches our target condition (win or loss)
                if non_miss_rewards[prev_trial_idx] != target_reward:
                    continue
                    
                # We found a trial that follows our target condition
                orig_curr_idx = filtered_to_orig[curr_trial_idx]
                
                # Skip if we don't have photometry data for this trial
                if orig_curr_idx not in valid_trials:
                    continue
                    
                # Get photometry data for this trial
                valid_idx = np.where(np.array(valid_trials) == orig_curr_idx)[0]
                if len(valid_idx) == 0:
                    continue
                    
                curr_photometry = epoched_data[valid_idx[0]]
                
                # Calculate average signal in the specified window
                window_signal = np.mean(curr_photometry[window_idx_start:window_idx_end])
                
                # Determine if the current choice is different from previous (switched)
                choice_switched = (non_miss_choices[prev_trial_idx] != non_miss_choices[curr_trial_idx])
                
                # Get state probabilities if available
                state_info = {'p_stochastic': None, 'p_leftbias': None, 'p_rightbias': None}
                if session_df is not None and i < len(session_df):
                    # Get state probabilities for this trial
                    state_info['p_stochastic'] = session_df.iloc[i].get('p_stochastic', None)
                    state_info['p_leftbias'] = session_df.iloc[i].get('p_leftbias', None) 
                    state_info['p_rightbias'] = session_df.iloc[i].get('p_rightbias', None)
                
                # Store the data
                all_trials_data.append(curr_photometry)
                all_trial_signals.append(window_signal)
                all_choice_switches.append(choice_switched)
                all_next_reward.append(non_miss_rewards[curr_trial_idx])  # Current trial's reward outcome
                all_state_probs.append(state_info)
                
        else:
            # POST-CUE ANALYSIS MODE
            # Look at current win/loss trials and analyze if the next choice switches
            
            for i in range(len(non_miss_rewards) - 1):  # Exclude the last trial (no next trial)
                curr_trial_idx = i
                next_trial_idx = i + 1
                
                # Skip if current trial doesn't match our target condition
                if non_miss_rewards[curr_trial_idx] != target_reward:
                    continue
                    
                # We found a trial matching our condition
                orig_curr_idx = filtered_to_orig[curr_trial_idx]
                
                # Skip if we don't have photometry data for this trial
                if orig_curr_idx not in valid_trials:
                    continue
                    
                # Get photometry data for this trial
                valid_idx = np.where(np.array(valid_trials) == orig_curr_idx)[0]
                if len(valid_idx) == 0:
                    continue
                    
                curr_photometry = epoched_data[valid_idx[0]]
                
                # Calculate average signal in the specified window
                window_signal = np.mean(curr_photometry[window_idx_start:window_idx_end])
                
                # Determine if the next choice is different from current (switched)
                choice_switched = (non_miss_choices[curr_trial_idx] != non_miss_choices[next_trial_idx])
                
                # Get state probabilities if available
                state_info = {'p_stochastic': None, 'p_leftbias': None, 'p_rightbias': None}
                if session_df is not None and i < len(session_df):
                    # Get state probabilities for this trial
                    state_info['p_stochastic'] = session_df.iloc[i].get('p_stochastic', None)
                    state_info['p_leftbias'] = session_df.iloc[i].get('p_leftbias', None) 
                    state_info['p_rightbias'] = session_df.iloc[i].get('p_rightbias', None)
                
                # Store the data
                all_trials_data.append(curr_photometry)
                all_trial_signals.append(window_signal)
                all_choice_switches.append(choice_switched)
                all_next_reward.append(non_miss_rewards[next_trial_idx])  # Next trial's reward outcome
                all_state_probs.append(state_info)
    
    # Check if we found any valid trials
    if len(all_trials_data) == 0:
        print(f"No valid {condition} trials found for analysis ({signal_window}) for {subject_id}")
        return None
        
    # Convert lists to numpy arrays
    all_trials_data = np.array(all_trials_data)
    all_trial_signals = np.array(all_trial_signals)
    all_choice_switches = np.array(all_choice_switches)
    all_next_reward = np.array(all_next_reward)
    
    # Sort trials into quartiles based on signal
    quartile_labels = pd.qcut(all_trial_signals, 4, labels=False)
    
    # Calculate switch rate for each quartile
    quartile_switch_rates = []
    quartile_trial_counts = []
    quartile_reward_rates = []
    
    # State threshold analysis per quartile
    quartile_state_counts = []
    state_threshold = 0.8  # Threshold for state assignment
    
    # Process each quartile
    for quartile in range(4):
        quartile_mask = quartile_labels == quartile
        trials_in_quartile = np.sum(quartile_mask)
        
        if trials_in_quartile > 0:
            # Calculate switch rate (% of trials where choice switched)
            switch_count = np.sum(all_choice_switches[quartile_mask])
            switch_rate = (switch_count / trials_in_quartile) * 100
            
            # Calculate reward rate for the relevant trial
            reward_rate = (np.sum(all_next_reward[quartile_mask]) / trials_in_quartile) * 100
            
            quartile_switch_rates.append(switch_rate)
            quartile_trial_counts.append(trials_in_quartile)
            quartile_reward_rates.append(reward_rate)
            
            # Calculate state proportions for this quartile
            quartile_states = {'stochastic': 0, 'biased': 0, 'uncertain': 0}
            quartile_indices = np.where(quartile_mask)[0]
            
            for idx in quartile_indices:
                state_info = all_state_probs[idx]
                p_stoch = state_info['p_stochastic']
                p_left = state_info['p_leftbias']
                p_right = state_info['p_rightbias']
                
                # Skip trials with missing state information
                if p_stoch is None or p_left is None or p_right is None:
                    quartile_states['uncertain'] += 1
                    continue
                
                # Assign state based on threshold
                if p_stoch >= state_threshold:
                    quartile_states['stochastic'] += 1
                elif p_left >= state_threshold or p_right >= state_threshold:
                    quartile_states['biased'] += 1
                else:
                    quartile_states['uncertain'] += 1
            
            quartile_state_counts.append(quartile_states)
        else:
            quartile_switch_rates.append(0)
            quartile_trial_counts.append(0)
            quartile_reward_rates.append(0)
            quartile_state_counts.append({'stochastic': 0, 'biased': 0, 'uncertain': 0})
            
    # Print results
    outcome_label = "Win" if condition == 'win' else "Loss"
    print(f"\n=== Analysis of {signal_window} Signal Quartiles ({outcome_label} trials): {subject_id} ===")
    print(f"Time window: ({window_start}s to {window_end}s)")
    print(f"Total trials analyzed: {len(all_trial_signals)}")
    
    if is_pre_cue_analysis:
        print(f"\nChoice Switch Rates by Pre-Cue Signal Quartile (after {outcome_label.lower()} trials):")
        print(f"(% where T0 choice differs from previous T-1 {outcome_label.lower()} trial)")
    else:
        print(f"\nChoice Switch Rates by Post-Cue Signal Quartile (for {outcome_label.lower()} trials):")
        print(f"(% where next T+1 choice differs from current T0 {outcome_label.lower()} trial)")
        
    for quartile in range(4):
        print(f"Quartile {quartile + 1}: {quartile_switch_rates[quartile]:.1f}% switch rate "
              f"({quartile_trial_counts[quartile]} trials)")
              
    print("\nReward Rates by Signal Quartile:")
    for quartile in range(4):
        print(f"Quartile {quartile + 1}: {quartile_reward_rates[quartile]:.1f}% rewarded")
    
    # Print state distribution by quartile
    print("\nState Distribution by Signal Quartile (threshold = 0.8):")
    for quartile in range(4):
        states = quartile_state_counts[quartile]
        total = quartile_trial_counts[quartile]
        
        if total > 0:
            # Calculate percentages
            stoch_pct = (states['stochastic'] / total) * 100
            biased_pct = (states['biased'] / total) * 100
            uncertain_pct = (states['uncertain'] / total) * 100
            
            print(f"Quartile {quartile + 1}: "
                  f"Stochastic: {states['stochastic']} ({stoch_pct:.1f}%), "
                  f"Biased: {states['biased']} ({biased_pct:.1f}%), "
                  f"Uncertain: {states['uncertain']} ({uncertain_pct:.1f}%)")
        else:
            print(f"Quartile {quartile + 1}: No trials")
    
    # Create verification plot if requested
    if plot_verification:
        # Create a single figure for just the photometry plot (no subplots needed)
        plt.figure(figsize=(12, 7))
        
        # Define colors for quartiles
        colors = ['blue', 'green', 'orange', 'red']  # Colors for quartiles
        
        # Plot each quartile's average trace
        for quartile in range(4):
            quartile_mask = quartile_labels == quartile
            if np.sum(quartile_mask) > 0:
                quartile_data = all_trials_data[quartile_mask]
                quartile_avg = np.mean(quartile_data, axis=0)
                quartile_sem = calculate_sem(quartile_data, axis=0)
                
                plt.fill_between(time_axis,
                               quartile_avg - quartile_sem,
                               quartile_avg + quartile_sem,
                               color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, quartile_avg,
                       color=colors[quartile], linewidth=2,
                       label=f'Quartile {quartile+1} (n={quartile_trial_counts[quartile]})')
                       
        # Highlight the time window used for sorting
        plt.axvspan(window_start, window_end, color='gray', alpha=0.3, label='Sorting Window')
        
        # Add reference lines
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('ΔF/F', fontsize=12)
        
        if is_pre_cue_analysis:
            plt.title(f'Trials After {outcome_label} Sorted by Pre-Cue Signal: {subject_id}', fontsize=14)
        else:
            plt.title(f'{outcome_label} Trials Sorted by {signal_window} Signal: {subject_id}', fontsize=14)
            
        plt.legend(loc='upper right')
        plt.xlim([-pre_cue_time, post_cue_time])
        
        # Add total trials information as text at the bottom of the plot
        plt.figtext(0.5, 0.01, f"Total trials analyzed: {len(all_trial_signals)}", 
                   ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        # Save the figure
        condition_str = condition  # 'win' or 'loss'
        mode_suffix = f"after_{condition_str}" if is_pre_cue_analysis else f"{condition_str}_trials"
        save_figure(plt.gcf(), subject_id, "pooled", f"{mode_suffix}_{signal_window}_quartiles")
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom
        plt.show()

    # Add statistical analysis if we have enough data
    if len(all_trial_signals) >= 100:  # Only perform stats with sufficient data
        add_statistical_test(subject_id, quartile_labels, all_choice_switches,
                             quartile_trial_counts, quartile_switch_rates)
        
    # Return analysis results
    return {
        'subject_id': subject_id,
        'time_axis': time_axis,
        'signal_window': signal_window,
        'condition': condition,
        'window_bounds': (window_start, window_end),
        'all_trials_data': all_trials_data,
        'all_trial_signals': all_trial_signals,
        'quartile_labels': quartile_labels,
        'quartile_switch_rates': quartile_switch_rates,
        'quartile_trial_counts': quartile_trial_counts,
        'quartile_reward_rates': quartile_reward_rates,
        'is_pre_cue_analysis': is_pre_cue_analysis
    }

def analyze_switch_probabilities(subject_id, session_date=None, behavior_df=None):
    """
    Analyze probability of switching choices following win vs loss trials 
    for a subject, and test if there is a significant difference.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
        
    Returns:
    --------
    dict: Analysis results including switch probabilities and statistical tests
    """
    # Store results
    results = {
        'subject_id': subject_id,
        'after_win': {'switches': 0, 'total': 0, 'rate': 0},
        'after_loss': {'switches': 0, 'total': 0, 'rate': 0}, 
        'sessions_analyzed': 0,
        'session_results': []
    }
    
    # Determine which sessions to analyze
    if session_date is None:
        if behavior_df is not None:
            # Extract sessions for this subject from behavior_df
            sessions = sorted(behavior_df[behavior_df['subjid'] == subject_id]['date'].unique())
        else:
            # Get all sessions from the file system
            subject_path = os.path.join(base_dir, subject_id)
            sessions = sorted([d for d in os.listdir(subject_path)
                            if os.path.isdir(os.path.join(subject_path, d)) and
                            os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])
    else:
        sessions = [session_date]
    
    print(f"Analyzing switch probabilities for {subject_id} across {len(sessions)} sessions...")
    
    # Process each session
    for sess in sessions:
        print(f"Processing {subject_id}/{sess}...")
        session_result = process_session(subject_id, sess, behavior_df=behavior_df)
        if not session_result:
            continue
            
        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])
        
        # Skip sessions with too few trials
        if len(choices) < 10:  # Need reasonable number of trials
            print(f"Skipping {subject_id}/{sess} - too few trials")
            continue
            
        # Filter out missed trials
        non_miss_mask = choices != 'M'
        filtered_choices = choices[non_miss_mask]
        filtered_rewards = rewards[non_miss_mask]
        
        # Skip if not enough trials after filtering
        if len(filtered_choices) < 10:
            print(f"Skipping {subject_id}/{sess} - too few non-missed trials")
            continue
            
        # Initialize counters for this session
        session_stats = {
            'session_date': sess,
            'after_win': {'switches': 0, 'total': 0, 'rate': 0},
            'after_loss': {'switches': 0, 'total': 0, 'rate': 0}
        }
        
        # Count switches after wins and losses
        for i in range(1, len(filtered_choices)):
            prev_outcome = filtered_rewards[i-1]
            curr_choice = filtered_choices[i]
            prev_choice = filtered_choices[i-1]
            
            # Check if choice switched
            switched = curr_choice != prev_choice
            
            # Categorize by previous trial outcome
            if prev_outcome == 1:  # Win
                session_stats['after_win']['total'] += 1
                if switched:
                    session_stats['after_win']['switches'] += 1
            else:  # Loss
                session_stats['after_loss']['total'] += 1
                if switched:
                    session_stats['after_loss']['switches'] += 1
        
        # Calculate switch rates for this session
        if session_stats['after_win']['total'] > 0:
            session_stats['after_win']['rate'] = (session_stats['after_win']['switches'] / session_stats['after_win']['total'] * 100)
        if session_stats['after_loss']['total'] > 0:
            session_stats['after_loss']['rate'] = (session_stats['after_loss']['switches'] / 
                                                 session_stats['after_loss']['total'] * 100)
        
        # Add session data to overall results
        results['session_results'].append(session_stats)
        results['after_win']['switches'] += session_stats['after_win']['switches']
        results['after_win']['total'] += session_stats['after_win']['total']
        results['after_loss']['switches'] += session_stats['after_loss']['switches']
        results['after_loss']['total'] += session_stats['after_loss']['total']
        results['sessions_analyzed'] += 1
        
        # Print session results
        print(f"  Session {sess} results:")
        print(f"    After win: {session_stats['after_win']['switches']}/{session_stats['after_win']['total']} " +
              f"switches ({session_stats['after_win']['rate']:.1f}%)")
        print(f"    After loss: {session_stats['after_loss']['switches']}/{session_stats['after_loss']['total']} " +
              f"switches ({session_stats['after_loss']['rate']:.1f}%)")
    
    # Calculate overall switch rates
    if results['after_win']['total'] > 0:
        results['after_win']['rate'] = results['after_win']['switches'] / results['after_win']['total'] * 100
    if results['after_loss']['total'] > 0:
        results['after_loss']['rate'] = results['after_loss']['switches'] / results['after_loss']['total'] * 100
    
    # Statistical testing: Chi-square test for independence
    contingency_table = np.array([
        [results['after_win']['switches'], results['after_win']['total'] - results['after_win']['switches']],
        [results['after_loss']['switches'], results['after_loss']['total'] - results['after_loss']['switches']]
    ])
    
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
        results['statistics'] = {
            'chi2': chi2,
            'p_value': p_chi2,
            'dof': dof,
            'expected': expected
        }
        
        # Fisher's exact test (more appropriate for smaller samples)
        odds_ratio, p_fisher = fisher_exact(contingency_table)
        results['statistics']['fisher_p_value'] = p_fisher
        results['statistics']['odds_ratio'] = odds_ratio
        
    except Exception as e:
        print(f"Error in statistical test: {e}")
        results['statistics'] = {'error': str(e)}
    
    # Print overall results with statistical test
    print("\n=== Overall Switch Probability Analysis ===")
    print(f"Subject: {subject_id}")
    print(f"Sessions analyzed: {results['sessions_analyzed']}")
    print(f"After win: {results['after_win']['switches']}/{results['after_win']['total']} " +
          f"switches ({results['after_win']['rate']:.1f}%)")
    print(f"After loss: {results['after_loss']['switches']}/{results['after_loss']['total']} " +
          f"switches ({results['after_loss']['rate']:.1f}%)")
    
    # Print statistical test results
    if 'statistics' in results and 'error' not in results['statistics']:
        print("\nStatistical Tests:")
        print(f"Chi-square test: χ² = {results['statistics']['chi2']:.2f}, p = {results['statistics']['p_value']:.5f}")
        print(f"Fisher's exact test: p = {results['statistics']['fisher_p_value']:.5f}, " + 
              f"odds ratio = {results['statistics']['odds_ratio']:.2f}")
        
        sig_level = 0.05
        if results['statistics']['p_value'] < sig_level or results['statistics']['fisher_p_value'] < sig_level:
            # Determine which condition has higher switch rate
            if results['after_win']['rate'] > results['after_loss']['rate']:
                print("\nResult: Subject significantly more likely to switch after wins than losses")
            else:
                print("\nResult: Subject significantly more likely to switch after losses than wins")
        else:
            print("\nResult: No significant difference in switch behavior after wins vs losses")
    
    # Create visualization
    if results['sessions_analyzed'] > 0:
        plt.figure(figsize=(12, 6))
        
        # If analyzing multiple sessions, plot session-by-session rates
        if len(results['session_results']) > 1:
            # Extract data for plotting
            sessions = list(range(1, results['sessions_analyzed'] + 1))
            win_rates = [s['after_win']['rate'] for s in results['session_results']]
            loss_rates = [s['after_loss']['rate'] for s in results['session_results']]
            
            # Plot session data
            plt.subplot(1, 2, 1)
            plt.plot(sessions, win_rates, 'o-', color='green', label='After Win')
            plt.plot(sessions, loss_rates, 'o-', color='red', label='After Loss')
            plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('Session Number')
            plt.ylabel('Switch Rate (%)')
            plt.title('Switch Rates Across Sessions')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Bar plot of overall rates on right side
            plt.subplot(1, 2, 2)
            
        # Create bar chart (standalone for single session, or right panel for multiple)
        bars = plt.bar([1, 2], 
                      [results['after_win']['rate'], results['after_loss']['rate']], 
                      color=['green', 'red'])
        
        # Add labels to bars
        for bar, count, total in zip(bars, 
                                   [results['after_win']['switches'], results['after_loss']['switches']],
                                   [results['after_win']['total'], results['after_loss']['total']]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{count}/{total}\n({height:.1f}%)', ha='center', va='bottom', fontsize=10)
        
        plt.xticks([1, 2], ['After Win', 'After Loss'])
        plt.ylabel('Switch Rate (%)')
        plt.title('Overall Switch Probabilities')
        plt.ylim(0, 100)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add statistical annotation if we have statistics
        if 'statistics' in results and 'p_value' in results['statistics']:
            p_val = min(results['statistics']['p_value'], results['statistics']['fisher_p_value'])
            if p_val < 0.001:
                sig_text = "***"
            elif p_val < 0.01:
                sig_text = "**"
            elif p_val < 0.05:
                sig_text = "*"
            else:
                sig_text = "n.s."
                
            # Draw a line connecting the bars with significance
            height = max(results['after_win']['rate'], results['after_loss']['rate']) + 5
            plt.plot([1, 2], [height, height], 'k-', linewidth=1.5)
            plt.text(1.5, height + 2, sig_text, ha='center', va='bottom', fontsize=14)
            
            # Add p-value text
            plt.text(1.5, height + 8, f"p = {p_val:.4f}", ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the figure
        save_figure(plt.gcf(), subject_id, "pooled", "switch_probabilities")
        plt.show()
    
    return results


def analyze_switch_probability_quartiles(subject_id, session_date=None, win_loss=False, behavior_df=None):
    """
    Analyze photometry signals binned by switch probability quartiles for a single session or pooled across sessions.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    win_loss : bool, optional
        Whether to split by rewarded/unrewarded trials

    Returns:
    --------
    dict: Analysis results including quartile bins and switch probability values
    """
    original_session_date = session_date

    all_plotting_data = []
    all_switch_probs = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
        sessions = sorted([d for d in os.listdir(subject_path)
                           if os.path.isdir(os.path.join(subject_path, d)) and
                           os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

        # Process each session separately
        for session_date in sessions:
            session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(
                    f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Calculate switch probabilities for this session
            behavior_data = session_result['behavioral_data']
            choices = np.array(behavior_data['choice'])

            # Filter out missed trials
            non_miss_mask = choices != 'M'
            filtered_choices = choices[non_miss_mask]

            # Skip if not enough choices after filtering
            if len(filtered_choices) < 10:
                print(f"Skipping {subject_id}/{session_date}, insufficient non-missed trials.")
                continue

            # Calculate moving average switch probability with window size 15
            window_size = 15
            switch_probs = []
            overall_switch_rate = 0

            # First calculate overall switch rate for the session
            for i in range(1, len(filtered_choices)):
                if filtered_choices[i] != filtered_choices[i - 1]:
                    overall_switch_rate += 1
            if len(filtered_choices) > 1:
                overall_switch_rate /= (len(filtered_choices) - 1)

            # Calculate moving average switch probabilities
            for i in range(len(filtered_choices)):
                if i < window_size:
                    # For early trials, use available data plus weighted overall rate
                    switches = 0
                    comparisons = 0

                    for j in range(1, i + 1):
                        if filtered_choices[j] != filtered_choices[j - 1]:
                            switches += 1
                        comparisons += 1

                    if comparisons > 0:
                        recent_rate = switches / comparisons
                    else:
                        recent_rate = overall_switch_rate

                    # Weight: how much of the window size we have available
                    available_weight = comparisons / window_size
                    switch_prob = (available_weight * recent_rate +
                                   (1 - available_weight) * overall_switch_rate)
                else:
                    # For later trials, use full window
                    switches = 0
                    for j in range(i - window_size + 1, i + 1):
                        if j > 0 and filtered_choices[j] != filtered_choices[j - 1]:
                            switches += 1
                    switch_prob = switches / window_size

                switch_probs.append(switch_prob)

            switch_probs = np.array(switch_probs)

            # Map these probabilities to the valid trials with photometry data
            non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                      if idx in session_result["non_m_trials"]])

            # Get the original indices for non-missed trials
            orig_non_miss_indices = np.where(non_miss_mask)[0]

            # Create mapping from valid photometry trial to behavior index
            trial_switch_probs = []
            for idx in non_m_indices:
                # Find the behavior index for this photometry trial
                if idx < len(orig_non_miss_indices):
                    behavior_idx = orig_non_miss_indices[idx]
                    if behavior_idx < len(switch_probs):
                        trial_switch_probs.append(switch_probs[behavior_idx])
                    else:
                        trial_switch_probs.append(overall_switch_rate)
                else:
                    trial_switch_probs.append(overall_switch_rate)

            # Store data for this session
            all_plotting_data.append(session_result['plotting_data'])
            all_switch_probs.extend(trial_switch_probs)
            all_reward_outcomes.append(session_result["reward_outcomes"][non_m_indices])
            time_axis = session_result['time_axis']

        plot_title = f'Pooled Photometry by Switch Probability Quartiles: {subject_id}'
        plotting_data = np.vstack(all_plotting_data)
        switch_probs = np.array(all_switch_probs)
        reward_outcomes = np.concatenate(all_reward_outcomes)

    else:
        # Single session analysis
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        # Calculate switch probabilities for this session
        behavior_data = session_result['behavioral_data']
        choices = np.array(behavior_data['choice'])

        # Filter out missed trials
        non_miss_mask = choices != 'M'
        filtered_choices = choices[non_miss_mask]

        # Skip if not enough choices after filtering
        if len(filtered_choices) < 10:
            print(f"Insufficient non-missed trials in {subject_id}/{session_date}")
            return None

        # Calculate moving average switch probability with window size 15
        window_size = 15
        switch_probs = []
        overall_switch_rate = 0

        # First calculate overall switch rate for the session
        for i in range(1, len(filtered_choices)):
            if filtered_choices[i] != filtered_choices[i - 1]:
                overall_switch_rate += 1
        if len(filtered_choices) > 1:
            overall_switch_rate /= (len(filtered_choices) - 1)

        # Calculate moving average switch probabilities
        for i in range(len(filtered_choices)):
            if i < window_size:
                # For early trials, use available data plus weighted overall rate
                switches = 0
                comparisons = 0

                for j in range(1, i + 1):
                    if filtered_choices[j] != filtered_choices[j - 1]:
                        switches += 1
                    comparisons += 1

                if comparisons > 0:
                    recent_rate = switches / comparisons
                else:
                    recent_rate = overall_switch_rate

                # Weight: how much of the window size we have available
                available_weight = comparisons / window_size
                switch_prob = (available_weight * recent_rate +
                               (1 - available_weight) * overall_switch_rate)
            else:
                # For later trials, use full window
                switches = 0
                for j in range(i - window_size + 1, i + 1):
                    if j > 0 and filtered_choices[j] != filtered_choices[j - 1]:
                        switches += 1
                switch_prob = switches / window_size

            switch_probs.append(switch_prob)

        switch_probs = np.array(switch_probs)

        # Map these probabilities to the valid trials with photometry data
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])

        # Get the original indices for non-missed trials
        orig_non_miss_indices = np.where(non_miss_mask)[0]

        # Create mapping from valid photometry trial to behavior index
        trial_switch_probs = []
        for idx in non_m_indices:
            # Find the behavior index for this photometry trial
            if idx < len(orig_non_miss_indices):
                behavior_idx = orig_non_miss_indices[idx]
                if behavior_idx < len(switch_probs):
                    trial_switch_probs.append(switch_probs[behavior_idx])
                else:
                    trial_switch_probs.append(overall_switch_rate)
            else:
                trial_switch_probs.append(overall_switch_rate)

        plotting_data = session_result['plotting_data']
        switch_probs = np.array(trial_switch_probs)
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        time_axis = session_result['time_axis']
        plot_title = f'LC Signal by Switch Probability Quartiles: {subject_id} - {session_date}'

    # Create quartile bins based on switch probabilities
    quartile_bins = pd.qcut(switch_probs, q=4, labels=False)

    # Calculate average switch probability for each quartile
    quartile_averages = []
    for quartile in range(4):
        quartile_avg = np.mean(switch_probs[quartile_bins == quartile])
        quartile_averages.append(quartile_avg)

    # Print average switch probabilities for each quartile
    print(f"\nAverage switch probabilities by quartile:")
    for quartile in range(4):
        print(f"Quartile {quartile + 1}: {quartile_averages[quartile]:.4f}")

    # Create the plot
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'red']  # From lowest to highest switch probability

    if win_loss:
        for quartile in range(4):
            quartile_rewarded = (quartile_bins == quartile) & (reward_outcomes == 1)
            quartile_unrewarded = (quartile_bins == quartile) & (reward_outcomes == 0)

            if np.sum(quartile_rewarded) > 0:
                rewarded_avg = np.mean(plotting_data[quartile_rewarded], axis=0)
                rewarded_sem = calculate_sem(plotting_data[quartile_rewarded], axis=0)
                plt.fill_between(time_axis,
                                 rewarded_avg - rewarded_sem,
                                 rewarded_avg + rewarded_sem,
                                 color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, rewarded_avg,
                         color=colors[quartile], linewidth=2,
                         label=f'Quartile {quartile + 1} Rewarded (n={np.sum(quartile_rewarded)})')

            if np.sum(quartile_unrewarded) > 0:
                unrewarded_avg = np.mean(plotting_data[quartile_unrewarded], axis=0)
                unrewarded_sem = calculate_sem(plotting_data[quartile_unrewarded], axis=0)
                plt.plot(time_axis, unrewarded_avg,
                         color=colors[quartile], linewidth=2, linestyle='--',
                         label=f'Quartile {quartile + 1} Unrewarded (n={np.sum(quartile_unrewarded)})')
    else:
        for quartile in range(4):
            quartile_trials = quartile_bins == quartile
            if np.sum(quartile_trials) > 0:
                quartile_avg = np.mean(plotting_data[quartile_trials], axis=0)
                quartile_sem = calculate_sem(plotting_data[quartile_trials], axis=0)

                plt.fill_between(time_axis,
                                 quartile_avg - quartile_sem,
                                 quartile_avg + quartile_sem,
                                 color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, quartile_avg,
                         color=colors[quartile], linewidth=2,
                         label=f'Quartile {quartile + 1} (n={np.sum(quartile_trials)})')

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')

    # Add text with quartile averages at the bottom of the plot
    quartile_text = "Average switch probabilities: " + ", ".join(
        [f"Q{q + 1}: {avg:.4f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom

    fig_name = "switch_probability_quartiles"
    if original_session_date is None:
        save_figure(plt.gcf(), subject_id, "pooled",
                f"{fig_name}{'_pooled'}{'_winloss' if win_loss else ''}")
    else:
        save_figure(plt.gcf(), subject_id, original_session_date,
                f"{fig_name}{'_winloss' if win_loss else ''}")

    plt.show()