
def compute_reward(self, quad_state, quad_vel, collision, obstacles, goal, power_usage):
    """Compute reward"""
    fin = 0
    done = 0
    reward = 0

    if collision:
        reward -= 100  # Penalty for collision
        fin = 1
    else:
        # Calculate 3D Euclidean distance to the goal
        dist = np.linalg.norm(np.array(quad_state) - np.array(goal))
        diff = self.last_dist - dist

        # Dynamic scaling of rewards and penalties based on proximity to goal
        scale_factor = 10 if dist > 5 else 50 if dist > 1 else 100
        
        
        if self.last_dist != 0:
            reward += diff * scale_factor  # Reward for getting closer

        
        if diff < 0:
            reward += diff * scale_factor / 2  # Penalize for increasing distance

        # Reward for being very close to the destination
        if dist < 5:
            reward += 500  

        # Reward for reaching the goal
        if dist < 1:
            reward += 1000  
            fin = 1

        # Penalize proximity to obstacles
        min_dist_to_obstacles = min(np.linalg.norm(np.array(quad_state) - np.array(obs)) for obs in obstacles)
        if min_dist_to_obstacles < 2:  
            reward -= (2 - min_dist_to_obstacles) * 50  

        # Penalizing high speeds and accelerations
        speed = np.linalg.norm(quad_vel)
        reward -= speed * 0.1  
        
        # Penalize for sudden changes in speed
        acceleration = np.linalg.norm(self.last_vel - quad_vel)
        reward -= acceleration * 0.05  

        # Penalize frequent changes in speed or direction
        if np.linalg.norm(self.last_vel - quad_vel) > 1:
            reward -= 10  

        # Penalizing time spent moving to encourage faster completion
        reward -= 1 

        # Penalize for high energy usage
        reward -= power_usage * 0.1  

        # Reward for movement towards the goal
        goal_direction = np.array(goal) - np.array(quad_state)
        goal_direction /= np.linalg.norm(goal_direction)  
        velocity_direction = quad_vel / np.linalg.norm(quad_vel) if np.linalg.norm(quad_vel) > 0 else np.zeros_like(goal_direction)
        direction_reward = np.dot(goal_direction, velocity_direction)  
        reward += direction_reward * 10 

        # Update last distance and velocity
        self.last_dist = dist
        self.last_vel = quad_vel

    
    if reward <= -100 or fin == 1:
        done = 1
        time.sleep(1)

    return reward, done

