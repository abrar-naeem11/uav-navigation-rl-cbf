import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import pybullet as p

from AOATTS import AOATTS
from Test_the_trained_agent import TD3


def save_episode_trajectory(env, agent_trajectory, dynamic_trajectories, episode_num, info, save_dir='trajectory_images'):
    """Save trajectory plot immediately with current environment state"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Capture all current state data NOW before anything changes
    target_pos = env.target_pos[:2].copy()
    agent_final_pos = env.agent_pos[:2].copy()
    
    # Get all obstacle data
    static_obstacles = []
    dynamic_obstacles = []
    
    for oid in env.obstacle_ids:
        try:
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=env.CLIENT)
            shape_data = p.getCollisionShapeData(oid, -1, physicsClientId=env.CLIENT)
            if shape_data:
                radius = shape_data[0][3][0]
            else:
                radius = 0.75
            
            obs_data = {
                'pos': (pos[0], pos[1]),
                'radius': radius
            }
            
            if oid in env.dynamic_obs_ids:
                dynamic_obstacles.append(obs_data)
            else:
                static_obstacles.append(obs_data)
        except Exception as e:
            print(f"Warning: Could not get obstacle {oid}: {e}")
    
    # Now create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Set plot limits
    ax.set_xlim(0, env.w)
    ax.set_ylim(0, env.h)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    # Title with episode info
    title = f'Episode {episode_num + 1}'
    if info.get('success', False):
        title += ' - SUCCESS'
    elif info.get('termination'):
        title += f' - {info["termination"].upper()}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Draw static obstacles (black circles)
    for obs in static_obstacles:
        circle = plt.Circle(obs['pos'], obs['radius'], 
                          color='black', alpha=0.6, zorder=2)
        ax.add_patch(circle)
    
    # Draw dynamic obstacles (green circles) at final positions
    for obs in dynamic_obstacles:
        circle = plt.Circle(obs['pos'], obs['radius'], 
                          color='green', alpha=0.6, zorder=2)
        ax.add_patch(circle)
    
    # Draw dynamic obstacle trajectories (green lines)
    for idx, trajectory in enumerate(dynamic_trajectories.values()):
        if len(trajectory) > 1:
            trajectory_array = np.array(trajectory)
            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                   color='green', linewidth=1.5, alpha=0.5, zorder=3, 
                   label='Dynamic Obstacle Path' if idx == 0 else '')
    
    # Draw agent trajectory (blue line)
    if len(agent_trajectory) > 1:
        trajectory_array = np.array(agent_trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
               color='blue', linewidth=2, alpha=0.7, zorder=4, label='Agent Path')
        
        # Add arrows to show direction
        num_arrows = min(5, max(1, len(trajectory_array) // 20))
        if num_arrows > 0 and len(trajectory_array) > 1:
            arrow_indices = np.linspace(0, len(trajectory_array) - 2, num_arrows, dtype=int)
            for idx in arrow_indices:
                dx = trajectory_array[idx + 1, 0] - trajectory_array[idx, 0]
                dy = trajectory_array[idx + 1, 1] - trajectory_array[idx, 1]
                if dx != 0 or dy != 0:
                    norm = np.sqrt(dx**2 + dy**2)
                    if norm > 0.1:  # Only draw if movement is significant
                        ax.arrow(trajectory_array[idx, 0], trajectory_array[idx, 1], 
                                dx * 0.8, dy * 0.8, head_width=1, head_length=0.8, 
                                fc='blue', ec='blue', alpha=0.6, zorder=4)
    
    # Draw target (red circle)
    target_circle = plt.Circle(target_pos, 0.5, 
                              color='red', alpha=0.8, zorder=5, label='Target')
    ax.add_patch(target_circle)
    
    # Draw agent final position (blue circle)
    agent_circle = plt.Circle(agent_final_pos, 0.5, 
                             color='blue', alpha=0.8, zorder=5, label='Agent Final')
    ax.add_patch(agent_circle)
    
    # Draw starting position of agent (cyan square)
    if len(agent_trajectory) > 0:
        start_pos = agent_trajectory[0]
        start_square = plt.Rectangle((start_pos[0] - 0.4, start_pos[1] - 0.4), 0.8, 0.8,
                                    color='cyan', alpha=0.7, zorder=5, label='Agent Start')
        ax.add_patch(start_square)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    
    # Add statistics text box
    stats_text = f'Steps: {len(agent_trajectory)}\n'
    stats_text += f'Static Obstacles: {len(static_obstacles)}\n'
    stats_text += f'Dynamic Obstacles: {len(dynamic_obstacles)}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save figure
    filename = f'{save_dir}/trajectory_episode_{episode_num + 1:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
    print(f"✓ Saved: {filename}")
    return filename


# Main evaluation script with trajectory visualization
def evaluate_with_visualization(model_path="td3_aoatt_new963.zip", num_episodes=100, gui=False):
    """Evaluate model and save trajectory visualizations"""
    
    env = AOATTS(gui=gui)
    model = TD3.load(model_path, env=env)
    
    # Statistics tracking
    success = 0
    collision = 0
    out_of_bound = 0
    timeout = 0
    qp_call = 0
    qp_success = 0
    
    records = []
    
    for episode in range(num_episodes):
        # Storage for this episode
        agent_trajectory = []
        dynamic_trajectories = {}
        
        obs, info = env.reset()
        
        # Record initial positions
        agent_trajectory.append(env.agent_pos[:2].copy())
        for oid in env.dynamic_obs_ids:
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=env.CLIENT)
            dynamic_trajectories[oid] = [pos[:2]]
        
        done, truncated = False, False
        step_count = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Record positions after each step
            agent_trajectory.append(env.agent_pos[:2].copy())
            for oid in env.dynamic_obs_ids:
                pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=env.CLIENT)
                dynamic_trajectories[oid].append(pos[:2])
        
        # IMMEDIATELY save trajectory plot BEFORE moving to next episode
        print(f"\nEpisode {episode + 1} complete: {info}, Steps: {step_count}")
        try:
            save_episode_trajectory(env, agent_trajectory, dynamic_trajectories, episode, info)
        except Exception as e:
            print(f"ERROR saving trajectory for episode {episode + 1}: {e}")
            import traceback
            traceback.print_exc()
        
        # Record statistics
        episode_record = info.copy()
        episode_record["episode"] = episode + 1
        episode_record["length"] = step_count
        episode_record["qp_call"] = env.qp_call_count
        episode_record["qp_success"] = env.qp_success_count
        episode_record["clf_mode"] = env.clf_mode_count
        episode_record["cbf_mode"] = env.cbf_mode_count
        records.append(episode_record)
        
        qp_call += env.qp_call_count
        qp_success += env.qp_success_count
        
        if info.get("termination") == "collision":
            collision += 1
        elif info.get("termination") == "out of bound":
            out_of_bound += 1
        elif info.get("termination") == "timeout":
            timeout += 1
        
        if done:
            success += 1
        
        print(f"QP Calls: {env.qp_call_count}, QP Success: {env.qp_success_count}, "
              f"CLF Mode: {env.clf_mode_count}, CBF Mode: {env.cbf_mode_count}")
    
    # Calculate and print final statistics
    success_rate = success / num_episodes
    collision_rate = collision / num_episodes
    out_of_bound_rate = out_of_bound / num_episodes
    timeout_rate = timeout / num_episodes
    qp_success_rate = qp_success / qp_call if qp_call > 0 else 0
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Episodes: {num_episodes}")
    print(f"Success Rate: {success_rate:.2%} ({success}/{num_episodes})")
    print(f"Collision Rate: {collision_rate:.2%} ({collision}/{num_episodes})")
    print(f"Out of Bound Rate: {out_of_bound_rate:.2%} ({out_of_bound}/{num_episodes})")
    print(f"Timeout Rate: {timeout_rate:.2%} ({timeout}/{num_episodes})")
    print(f"QP Success Rate: {qp_success_rate:.2%} ({qp_success}/{qp_call})")
    print("="*60)
    
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv("trajectory_evaluation_results.csv", index=False)
    print(f"\nResults saved to: trajectory_evaluation_results.csv")
    print(f"Trajectory images saved to: trajectory_images/")
    
    env.close()
    return df


if __name__ == "__main__":
    # Run evaluation with visualization
    # Set gui=True if you want to see the simulation in real-time (slower)
    # Set gui=False for faster evaluation (recommended)
    results_df = evaluate_with_visualization(
        model_path="td3_aoatt_new963.zip",
        num_episodes=100,
        gui=False
    )
