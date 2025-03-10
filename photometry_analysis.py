from random import choice

import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import glob
import warnings
import pandas as pd
from pytz.exceptions import NonExistentTimeError

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global parameters
base_dir = r"Z:\delab\lab-members\joanna\photometry\preprocess"
output_dir = r"Z:\delab\lab-members\joschua\photometry_analysis"
sampling_rate = 120
pre_cue_time = 3
post_cue_time = 5
pre_cue_samples = int(pre_cue_time * sampling_rate)
post_cue_samples = int(post_cue_time * sampling_rate)
total_window_samples = pre_cue_samples + post_cue_samples

PARQUET_PATH = r"Z:\delab\matchingpennies\matchingpennies_datatable.parquet"
CODE_VERSION = "1.0.4"  # Increment this when making analysis changes

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


def load_behavior_data(subject_id, session_date):
    """Load behavioral data from Parquet and filter it by subject and session date."""
    try:
        # Load the parquet file with pyarrow
        df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")

        df['date'] = df['date'].astype(str)  # Ensure date is a string
        session_data = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]
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


def process_session(subject_id, session_date, force_recompute=False):
    """Process a single session for a given subject"""

    if not force_recompute:
        saved_result = check_saved_results(subject_id, session_date)  
        if saved_result is not None:
            return saved_result

    full_dir = os.path.join(base_dir, subject_id, session_date)
    deltaff_file = os.path.join(full_dir, "deltaff.npy")
    pkl_file = find_pkl_file(full_dir)

    if not pkl_file or not os.path.exists(deltaff_file):
        print(f"Missing files for {subject_id}/{session_date}")
        return None

    try:
        deltaff_data = np.load(deltaff_file)
        print(f"{subject_id}/{session_date}: Loaded deltaff data with shape: {deltaff_data.shape}")

        with open(pkl_file, "rb") as f:
            pkl_data = pickle.load(f)
        print(f"{subject_id}/{session_date}: Loaded pkl data: {os.path.basename(pkl_file)}")

        # Extract pulse indices
        pulse_indices = pkl_data["pulse_inds_1"]
        num_trials = len(pulse_indices)
        print(f"{subject_id}/{session_date}: Found {num_trials} pulse indices")

        behavior_data = load_behavior_data(subject_id, session_date)
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
            'plotting_data': plotting_data
        }

        save_results(result, subject_id, session_date)
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

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('ΔF/F', fontsize=12)
    ax1.set_title(f'Photometry Response: {analysis_result["subject_id"]} - {analysis_result["session_date"]}',
                  fontsize=14)
    ax1.set_xlim([-pre_cue_time, post_cue_time])
    ax1.legend(loc='upper right')

    # Add statistics text box
    stats_text = (f"Trials: {len(analysis_result['non_m_trials'])} (excluding missed trials)\n"
                  f"Peak: {np.max(analysis_result['trial_average']):.4f}\n"
                  f"Baseline: {np.mean(analysis_result['trial_average'][:pre_cue_samples]):.4f}")
    ax1.text(-pre_cue_time + 0.2, np.max(analysis_result['trial_average']) * 0.9, stats_text,
             bbox=dict(facecolor='white', alpha=0.7))

    # Second subplot: Heatmap (if enabled)
    if show_heatmap:
        ax2 = axes[1]
        trial_data = analysis_result['plotting_data']
        im = ax2.imshow(trial_data,
                        aspect='auto',
                        extent=[-pre_cue_time, post_cue_time, 0, len(trial_data)],
                        cmap='viridis',
                        interpolation='nearest')

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Trial Number', fontsize=12)
        ax2.set_title('Trial-by-Trial Heatmap', fontsize=14)
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


def analyze_pooled_data(subject_id, win_loss=False, force_recompute=False, fig=None):
    """Analyze and visualize pooled data for a subject without loading from saved files"""
    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(12, 7))

    # Check for saved pooled results
    if not force_recompute:
        saved_results = check_saved_pooled_results(subject_id, win_loss)
        if saved_results is not None:
            # Recreate the figure from saved results
            plt.figure(figsize=(12, 7))

            if win_loss:
                # Rewarded data
                if saved_results.get('rewarded_avg') is not None:
                    rewarded_avg = saved_results['rewarded_avg']
                    rewarded_sem = saved_results['rewarded_sem']
                    plt.fill_between(saved_results['time_axis'],
                                    rewarded_avg - rewarded_sem,
                                    rewarded_avg + rewarded_sem,
                                    color='lightgreen', alpha=0.4, label='Rewarded ± SEM')
                    plt.plot(saved_results['time_axis'], rewarded_avg,
                            color='green', linewidth=2.5, label='Rewarded Avg')

                # Unrewarded data
                if saved_results.get('unrewarded_avg') is not None:
                    unrewarded_avg = saved_results['unrewarded_avg']
                    unrewarded_sem = saved_results['unrewarded_sem']
                    plt.fill_between(saved_results['time_axis'],
                                    unrewarded_avg - unrewarded_sem,
                                    unrewarded_avg + unrewarded_sem,
                                    color='lightsalmon', alpha=0.4, label='Unrewarded ± SEM')
                    plt.plot(saved_results['time_axis'], unrewarded_avg,
                            color='darkorange', linewidth=2.5, label='Unrewarded Avg')
            else:
                pooled_sem = saved_results['pooled_sem']
                plt.fill_between(saved_results['time_axis'],
                                 saved_results['pooled_average'] - pooled_sem,  
                                 saved_results['pooled_average'] + pooled_sem,  
                                 color='lightgreen', alpha=0.4,
                                 label='Mean ± SEM')  

                # Plot session averages
            num_sessions = len(saved_results['session_dates'])
            blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))

            for idx, session_avg in enumerate(saved_results.get('session_averages', [])):
                plt.plot(saved_results['time_axis'], session_avg,
                         alpha=0.6, linewidth=1, linestyle='-',
                         color=blue_colors[idx],
                         label=f"Session {saved_results['session_dates'][idx]}" if idx < 5 else "_nolegend_")

            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('ΔF/F', fontsize=12)
            plt.title(f'Pooled Photometry Response: {subject_id} ({len(saved_results["session_dates"])} sessions)',
                      fontsize=14)
            plt.xlim([-pre_cue_time, post_cue_time])

            # Recreate the stats text box
            stats_text = (f"Total Sessions: {len(saved_results['session_dates'])}\n"
                          f"Total Trials: {saved_results['total_trials']}\n"
                          f"Peak: {np.max(saved_results['pooled_average']):.4f}\n"
                          f"Baseline: {np.mean(saved_results['pooled_average'][:pre_cue_samples]):.4f}")
            plt.text(-pre_cue_time + 0.2, np.max(saved_results['pooled_average']) * 1.2, stats_text,
                     bbox=dict(facecolor='white', alpha=0.7))

            # Limit legend items if too many sessions
            if len(saved_results['session_dates']) > 5:
                handles, labels = plt.gca().get_legend_handles_labels()
                limited_handles = handles[:8]
                limited_labels = labels[:8]
                limited_labels.append(f"+ {len(saved_results['session_dates']) - 5} more sessions")
                plt.legend(limited_handles, limited_labels, loc='upper right', fontsize=10)
            else:
                plt.legend(loc='upper right', fontsize=10)

            plt.tight_layout()

            # Save figure
            save_figure(plt.gcf(), subject_id, "pooled",
                        f"pooled_results{'_winloss' if win_loss else ''}")

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

    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d)) and
                      os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))])
    
    num_sessions = len(sessions)
    blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))  # Create blue gradient

    for idx, session_date in enumerate(sessions):
        session_path = os.path.join(subject_dir, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            result = process_session(subject_id, session_date)
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

    # Create the pooled plot
    plt.figure(figsize=(12, 7))

    rewarded_avg = None
    unrewarded_avg = None

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
        rewarded_data = np.vstack(rewarded_data) if rewarded_data else np.array([])
        unrewarded_data = np.vstack(unrewarded_data) if unrewarded_data else np.array([])

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
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(f'Pooled Photometry Response: {subject_id} ({len(all_sessions)} sessions)', fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])

    # Limit legend items if too many sessions
    if len(all_sessions) > 5:
        handles, labels = plt.gca().get_legend_handles_labels()
        limited_handles = handles[:8]
        limited_labels = labels[:8]
        limited_labels.append(f"+ {len(all_sessions) - 5} more sessions")
        plt.legend(limited_handles, limited_labels, loc='upper right', fontsize=10)
    else:
        plt.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Add statistics - count only non-'M' trials
    total_trials = sum(len(session['non_m_trials']) for session in all_sessions)
    stats_text = (f"Total Sessions: {len(all_sessions)}\n"
                  f"Total Trials: {total_trials} (excluding 'M' choices)\n"
                  f"Peak: {np.max(pooled_average):.4f}\n"
                  f"Baseline: {np.mean(pooled_average[:pre_cue_samples]):.4f}")
    plt.text(-pre_cue_time + 0.2, pooled_average.max() * 1.2, stats_text,
             bbox=dict(facecolor='white', alpha=0.7))

    # Prepare pooled result
    pooled_result = {
        'code_version': CODE_VERSION,
        'subject_id': subject_id,
        'session_dates': session_dates,
        'pooled_data': pooled_data,
        'pooled_average': pooled_average,
        'pooled_sem': calculate_sem(pooled_data, axis=0),  
        'time_axis': time_axis,
        'total_trials': total_trials,
        'session_averages': session_averages,
        'rewarded_avg': rewarded_avg,
        'rewarded_sem': calculate_sem(rewarded_data, axis=0) if rewarded_data.size > 0 else None, 
        'unrewarded_avg': unrewarded_avg,
        'unrewarded_sem': calculate_sem(unrewarded_data, axis=0) if unrewarded_data.size > 0 else None 
    }

    # Save pooled results
    save_pooled_results(pooled_result, subject_id, win_loss)

    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", f"pooled_results{'_winloss' if win_loss else ''}")

    plt.show()
    return pooled_result


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


# Function aliases with clear names for Jupyter notebook cells
def analyze_specific_session(subject_id, session_date, show_heatmap=False, win_loss=False):
    """Analyze and visualize a specific session"""
    print(f"Analyzing session {subject_id}/{session_date}...")
    analysis_result = process_session(subject_id, session_date)
    if analysis_result:
        return plot_session_results(analysis_result, show_heatmap=show_heatmap, win_loss=win_loss)
    return None


def pooled_results(subject_id, win_loss=False):
    """Analyze and visualize pooled results for a subject"""
    print(f"Analyzing pooled results for subject {subject_id}...")
    return analyze_pooled_data(subject_id, win_loss=win_loss)


def all_results(win_loss=False, force_recompute=False):
    """Analyze and visualize results for all subjects"""
    print("Analyzing all subjects...")
    return analyze_all_subjects(win_loss=win_loss, force_recompute=force_recompute)


def analyze_reward_rate_quartiles(subject_id, session_date=None, win_loss=False):
    """
    Analyze photometry signals binned by reward rate quartiles for a single session or pooled across sessions
    """
    all_plotting_data = []
    all_reward_rates = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
        sessions = sorted([d for d in os.listdir(subject_path)
                         if os.path.isdir(os.path.join(subject_path, d)) and
                         os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])
        
        # Process each session separately to maintain session-specific reward rate context
        for session_date in sessions:
            session_result = process_session(subject_id, session_date)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Calculate reward rates for this session
            behavior_data = session_result['behavioral_data']
            rewards = np.array(behavior_data['reward'])
            window_size = max(int(len(rewards) * 0.1), 1)
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
        session_result = process_session(subject_id, session_date)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        window_size = max(int(len(rewards) * 0.1), 1)
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
        plot_title = f'Photometry by Reward Rate Quartiles: {subject_id} - {session_date}'

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
    quartile_text = "Average reward rates: " + ", ".join([f"Q{q+1}: {avg:.4f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom

    # Save the figure
    fig_name = f"reward_rate_quartiles{'_pooled' if session_date is None else ''}"
    save_figure(plt.gcf(), subject_id, session_date or "pooled", 
               f"{fig_name}{'_winloss' if win_loss else ''}")

    plt.show()

    return {
        'quartile_bins': quartile_bins,
        'reward_rates': reward_rates
    }


def analyze_comp_confidence_quartiles(subject_id, session_date=None, win_loss=False):
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

    Returns:
    --------
    dict: Analysis results including quartile bins and confidence values
    """
    all_plotting_data = []
    all_confidences = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
        sessions = sorted([d for d in os.listdir(subject_path)
                           if os.path.isdir(os.path.join(subject_path, d)) and
                           os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

        # Process each session separately to maintain session-specific confidence context
        for session_date in sessions:
            session_result = process_session(subject_id, session_date)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Get behavioral data for this session
            behavior_data = session_result['behavioral_data']

            # Load the parquet file to get min_pvalue data
            try:
                df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                df['date'] = df['date'].astype(str)  # Ensure date is a string
                session_data = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]

                if session_data.empty:
                    print(f"No p-value data found for {subject_id} on {session_date}")
                    continue

                # Extract p-values and calculate confidence
                p_values = session_data['min_pvalue'].values
                min_p_value = 1e-12
                p_values = np.maximum(p_values, min_p_value)
                confidence = -np.log10(p_values)

                # Calculate moving average confidence with window size 15
                window_size = 15
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
        session_result = process_session(subject_id, session_date)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        # Load the parquet file to get min_pvalue data
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            session_data = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]

            if session_data.empty:
                print(f"No p-value data found for {subject_id} on {session_date}")
                return None

            # Extract p-values and calculate confidence
            p_values = session_data['min_pvalue'].values
            min_p_value = 1e-12
            p_values = np.maximum(p_values, min_p_value)
            confidence = -np.log10(p_values)

            # Calculate moving average confidence with window size 15
            window_size = 15
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
            plot_title = f'Photometry by Computer Confidence Quartiles: {subject_id} - {session_date}'

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
    colors = ['blue', 'green', 'orange', 'red']  # From lowest to highest confidence

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
    quartile_text = "Average confidence values: " + ", ".join([f"Q{q+1}: {avg:.4f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom

    # Save the figure
    fig_name = f"computer_confidence_quartiles{'_pooled' if session_date is None else ''}"
    save_figure(plt.gcf(), subject_id, session_date or "pooled",
                f"{fig_name}{'_winloss' if win_loss else ''}")

    plt.show()

    return {
        'quartile_bins': quartile_bins,
        'confidence_rates': confidence_rates
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

def plot_per_session_win_loss(subject_id):
    """
    Plot win/loss traces for each session of a subject with choice history
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
        
    Returns:
    --------
    dict: Dictionary of session win-loss analyses
    """
    # Find all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

    # Find max peak for consistent y-axis scaling
    max_peak = float('-inf')
    min_peak = float('inf')
    session_analyses = {}

    # First pass: find maximum and minimum values for proper y-axis scaling
    for session_date in sessions:
        session_result = process_session(subject_id, session_date)
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

        # Compute average rewarded and unrewarded signals
        if len(rewarded_trials) > 0:
            rewarded_avg = np.mean(rewarded_trials, axis=0)
            rewarded_sem = calculate_sem(rewarded_trials, axis=0)
            max_peak = max(max_peak, np.max(rewarded_avg + rewarded_sem))
            min_peak = min(min_peak, np.min(rewarded_avg - rewarded_sem))
            
        if len(unrewarded_trials) > 0:
            unrewarded_avg = np.mean(unrewarded_trials, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)
            max_peak = max(max_peak, np.max(unrewarded_avg + unrewarded_sem))
            min_peak = min(min_peak, np.min(unrewarded_avg - unrewarded_sem))

    # Add 20% padding to y-axis limits
    y_range = max_peak - min_peak
    y_max = max_peak + 0.1 * y_range
    y_min = min_peak - 0.1 * y_range

    # Calculate number of sessions and rows/columns for subplots
    n_sessions = len(sessions)
    n_cols = 3  # Always 3 columns
    n_rows = (n_sessions + n_cols - 1) // n_cols  # Ceiling division for number of rows needed
    
    # Create figure with proper size
    fig = plt.figure(figsize=(18, 5*n_rows))  # Wider figure, height based on number of rows
    
    # Create GridSpec to control the layout - 2 rows per session (photometry + choice)
    gs = plt.GridSpec(n_rows*2, n_cols, height_ratios=[2, 1] * n_rows)  # Repeat [2,1] pattern for each row
    
    # Process each session
    for i, session_date in enumerate(sessions):
        row = i // n_cols  # Row index (integer division)
        col = i % n_cols   # Column index
        
        session_result = process_session(subject_id, session_date)
        if not session_result:
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

        # Create subplot for photometry data (top row for this session)
        ax_photo = fig.add_subplot(gs[row*2, col])
        
        # Photometry data - rewarded
        if len(rewarded_trials) > 0:
            rewarded_avg = np.mean(rewarded_trials, axis=0)
            rewarded_sem = calculate_sem(rewarded_trials, axis=0)
            ax_photo.fill_between(session_result['time_axis'],
                       rewarded_avg - rewarded_sem,
                       rewarded_avg + rewarded_sem,
                       color='lightgreen', alpha=0.3)
            ax_photo.plot(session_result['time_axis'], rewarded_avg,
                 color='green', linewidth=2,
                 label=f'Win (n={len(rewarded_trials)})')
        
        # Photometry data - unrewarded
        if len(unrewarded_trials) > 0:
            unrewarded_avg = np.mean(unrewarded_trials, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)
            ax_photo.fill_between(session_result['time_axis'],
                       unrewarded_avg - unrewarded_sem,
                       unrewarded_avg + unrewarded_sem,
                       color='lightsalmon', alpha=0.3)
            ax_photo.plot(session_result['time_axis'], unrewarded_avg,
                 color='darkorange', linewidth=2,
                 label=f'Loss (n={len(unrewarded_trials)})')
        
        # Add vertical line at cue onset
        ax_photo.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax_photo.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Set y-axis limit consistent across all photometry subplots
        ax_photo.set_ylim([y_min, y_max])
        
        # Labels and formatting for photometry subplot
        ax_photo.set_xlabel('Time (s)')
        ax_photo.set_ylabel('ΔF/F')
        ax_photo.set_title(f'Session {session_date} (n={len(rewarded_trials) + len(unrewarded_trials)} trials)')
        ax_photo.legend(loc='upper right')
        
        # Store analysis results
        session_analyses[session_date] = {
            'rewarded_avg': rewarded_avg if len(rewarded_trials) > 0 else None,
            'unrewarded_avg': unrewarded_avg if len(unrewarded_trials) > 0 else None,
            'rewarded_n': len(rewarded_trials),
            'unrewarded_n': len(unrewarded_trials),
            'time_axis': session_result['time_axis']
        }
        
        # Create subplot for choice history (bottom row)
        ax_choice = fig.add_subplot(gs[row*2 + 1, col])
        
        # Extract choices and rewards
        choices = session_result['behavioral_data']['choice']
        rewards = session_result['behavioral_data']['reward']
        
        # Plot choice history
        for j, choice in enumerate(choices):
            if choice == 'L':
                ax_choice.plot([j + 1, j + 1], [0, 1], 'r-', linewidth=1.5)
                if rewards[j] == 1:
                    ax_choice.plot(j + 1, 1, 'ro', markersize=8, fillstyle='none')
            elif choice == 'R':
                ax_choice.plot([j + 1, j + 1], [0, -1], 'b-', linewidth=1.5)
                if rewards[j] == 1:
                    ax_choice.plot(j + 1, -1, 'bo', markersize=8, fillstyle='none')
        
        # Add the middle line
        ax_choice.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Set the y-axis limits and labels
        ax_choice.set_ylim(-1.5, 1.5)
        ax_choice.set_yticks([-1, 0, 1])
        ax_choice.set_yticklabels(['Right', '', 'Left'])
        
        # Set the x-axis and title
        ax_choice.set_xlabel('Trial Number')
        ax_choice.set_title('Choice History')
        ax_choice.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, subject_id, "all_sessions", "per_session_win_loss_with_choices")
    
    plt.show()

def analyze_session_win_loss_difference_gap(subject_id, session_date=None):
    """
    Analyze win-loss difference across photometry sessions for a subject

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.

    Returns:
    --------
    dict: Dictionary of session win-loss difference analyses
    """
    # If no specific session provided, get all sessions for the subject
    if session_date is None:
        subject_path = os.path.join(base_dir, subject_id)
        sessions = sorted([d for d in os.listdir(subject_path)
                    if os.path.isdir(os.path.join(subject_path, d)) and
                    os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])
    else:
        sessions = [session_date]

    num_sessions = len(sessions)
    blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))

    # Prepare figure for win-loss difference plot
    plt.figure(figsize=(12, 7))

    # Store results for each session
    session_differences = {}

    # Process each session
    for idx, session_date in enumerate(sessions):
        # Process the session
        session_result = process_session(subject_id, session_date)
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

        # Store the difference
        session_differences[session_date] = win_loss_diff

        # Plot differences with SEM
        plt.fill_between(session_result['time_axis'],
                        win_loss_diff - win_loss_sem,
                        win_loss_diff + win_loss_sem,
                        color=blue_colors[idx], alpha=0.2)
        plt.plot(session_result['time_axis'], win_loss_diff,
                color=blue_colors[idx],
                label=f'Session {session_date}', linewidth=2)

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Rewarded - Unrewarded ΔF/F', fontsize=12)
    plt.title(f'Win-Loss Difference: {subject_id}', fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the figure
    save_figure(plt.gcf(), subject_id, "win_loss_diff", "win_loss_difference")

    plt.show()

    """
    Plot win/loss traces for each session of a subject with choice history
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
        
    Returns:
    --------
    dict: Dictionary of session win-loss analyses
    """
    # Find all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

    # Find max peak for consistent y-axis scaling
    max_peak = float('-inf')
    min_peak = float('inf')
    session_analyses = {}

    # First pass: find maximum and minimum values for proper y-axis scaling
    for session_date in sessions:
        session_result = process_session(subject_id, session_date)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
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

        # Compute average rewarded and unrewarded signals
        if len(rewarded_trials) > 0:
            rewarded_avg = np.mean(rewarded_trials, axis=0)
            rewarded_sem = calculate_sem(rewarded_trials, axis=0)
            max_peak = max(max_peak, np.max(rewarded_avg + rewarded_sem))
            min_peak = min(min_peak, np.min(rewarded_avg - rewarded_sem))
            
        if len(unrewarded_trials) > 0:
            unrewarded_avg = np.mean(unrewarded_trials, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)
            max_peak = max(max_peak, np.max(unrewarded_avg + unrewarded_sem))
            min_peak = min(min_peak, np.min(unrewarded_avg - unrewarded_sem))

    # Add 20% padding to y-axis limits
    y_range = max_peak - min_peak
    y_max = max_peak + 0.1 * y_range
    y_min = min_peak - 0.1 * y_range

    # Calculate number of sessions and columns for subplots
    n_sessions = len(sessions)
    n_cols = min(3, n_sessions)
    
    # Create figure with 2 rows (photometry on top, choice history below)
    fig = plt.figure(figsize=(n_cols * 6, 12))
    
    # Create GridSpec to control the layout
    gs = plt.GridSpec(2, n_cols, height_ratios=[2, 1])  # 2:1 ratio for photometry:choice plots
    
    # Process each session
    for i, session_date in enumerate(sessions):
        session_result = process_session(subject_id, session_date)
        if not session_result:
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

        # Create subplot for photometry data (top row)
        ax_photo = fig.add_subplot(gs[0, i % n_cols])
        
        # Photometry data - rewarded
        if len(rewarded_trials) > 0:
            rewarded_avg = np.mean(rewarded_trials, axis=0)
            rewarded_sem = calculate_sem(rewarded_trials, axis=0)
            ax_photo.fill_between(session_result['time_axis'],
                       rewarded_avg - rewarded_sem,
                       rewarded_avg + rewarded_sem,
                       color='lightgreen', alpha=0.3)
            ax_photo.plot(session_result['time_axis'], rewarded_avg,
                 color='green', linewidth=2,
                 label=f'Win (n={len(rewarded_trials)})')
        
        # Photometry data - unrewarded
        if len(unrewarded_trials) > 0:
            unrewarded_avg = np.mean(unrewarded_trials, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)
            ax_photo.fill_between(session_result['time_axis'],
                       unrewarded_avg - unrewarded_sem,
                       unrewarded_avg + unrewarded_sem,
                       color='lightsalmon', alpha=0.3)
            ax_photo.plot(session_result['time_axis'], unrewarded_avg,
                 color='darkorange', linewidth=2,
                 label=f'Loss (n={len(unrewarded_trials)})')
        
        # Add vertical line at cue onset
        ax_photo.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax_photo.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Set y-axis limit consistent across all photometry subplots
        ax_photo.set_ylim([y_min, y_max])
        
        # Labels and formatting for photometry subplot
        ax_photo.set_xlabel('Time (s)')
        ax_photo.set_ylabel('ΔF/F')
        ax_photo.set_title(f'Session {session_date} (n={len(rewarded_trials) + len(unrewarded_trials)} trials)')
        ax_photo.legend(loc='upper right')
        
        # Store analysis results
        session_analyses[session_date] = {
            'rewarded_avg': rewarded_avg if len(rewarded_trials) > 0 else None,
            'unrewarded_avg': unrewarded_avg if len(unrewarded_trials) > 0 else None,
            'rewarded_n': len(rewarded_trials),
            'unrewarded_n': len(unrewarded_trials),
            'time_axis': session_result['time_axis']
        }
        
        # Create subplot for choice history (bottom row)
        ax_choice = fig.add_subplot(gs[1, i % n_cols])
        
        # Extract choices and rewards
        choices = session_result['behavioral_data']['choice']
        rewards = session_result['behavioral_data']['reward']
        
        # Plot choice history
        for j, choice in enumerate(choices):
            if choice == 'L':
                ax_choice.plot([j + 1, j + 1], [0, 1], 'r-', linewidth=1.5)
                if rewards[j] == 1:
                    ax_choice.plot(j + 1, 1, 'ro', markersize=8, fillstyle='none')
            elif choice == 'R':
                ax_choice.plot([j + 1, j + 1], [0, -1], 'b-', linewidth=1.5)
                if rewards[j] == 1:
                    ax_choice.plot(j + 1, -1, 'bo', markersize=8, fillstyle='none')
        
        # Add the middle line
        ax_choice.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Set the y-axis limits and labels
        ax_choice.set_ylim(-1.5, 1.5)
        ax_choice.set_yticks([-1, 0, 1])
        ax_choice.set_yticklabels(['Right', '', 'Left'])
        
        # Set the x-axis and title
        ax_choice.set_xlabel('Trial Number')
        ax_choice.set_title('Choice History')
        ax_choice.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, subject_id, "all_sessions", "per_session_win_loss_with_choices")
    
    plt.show()

def analyze_previous_outcome_effect(subject_id):
    """
    Analyze photometry signals based on previous and current trial outcomes.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
        
    Returns:
    --------
    dict: Analysis results
    """
    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None

    # Process each session and collect results
    all_sessions = []
    all_plotting_data = []
    all_prev_rewards = []
    all_curr_rewards = []
    session_dates = []

    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d)) and
                      os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))])
    
    # Process sessions in chronological order
    for session_date in sessions:
        session_path = os.path.join(subject_dir, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            result = process_session(subject_id, session_date)
            if result:
                if len(result['non_m_trials']) < 100: 
                    print(f"Skipping {subject_id}/{session_date} due to low number of trials")
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
                
                # Store data
                all_plotting_data.append(session_plots)
                all_curr_rewards.append(curr_rewards)
                all_prev_rewards.append(prev_rewards)
    
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
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Define colors and labels
    colors = {
        'prev_win_curr_win': 'darkgreen',
        'prev_win_curr_loss': 'indianred', 
        'prev_loss_curr_win': 'mediumseagreen',
        'prev_loss_curr_loss': 'firebrick'
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
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Labels and formatting
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(f'Photometry by Previous Trial Outcome: {subject_id} ({len(all_sessions)} sessions)', fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", "previous_outcome_effect")
    
    plt.show()
    
    # Return analysis results
    return {
        'subject_id': subject_id,
        'session_dates': session_dates,
        'time_axis': time_axis,
        'condition_data': condition_data
    }