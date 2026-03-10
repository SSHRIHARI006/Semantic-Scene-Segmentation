import cv2
import numpy as np
import heapq
from typing import List, Tuple, Optional


class AStarPathPlanner:
    """A* path planning algorithm for grid-based navigation"""
    
    def __init__(self, obstacle_grid):
        """
        Initialize path planner
        
        Args:
            obstacle_grid: Binary grid where 1=obstacle, 0=navigable
        """
        self.grid = obstacle_grid
        self.height, self.width = obstacle_grid.shape
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-connected)"""
        x, y = pos
        neighbors = []
        
        # 8-connected grid (includes diagonals)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < self.height and 0 <= ny < self.width:
                # Check if navigable
                if self.grid[nx, ny] == 0:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path using A* algorithm
        
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            
        Returns:
            List of positions forming the path, or None if no path exists
        """
        # Ensure start and goal are valid
        if not (0 <= start[0] < self.height and 0 <= start[1] < self.width):
            return None
        if not (0 <= goal[0] < self.height and 0 <= goal[1] < self.width):
            return None
        
        # If start or goal is an obstacle, try to find nearest navigable cell
        if self.grid[start[0], start[1]] == 1:
            start = self._find_nearest_navigable(start)
            if start is None:
                return None
        
        if self.grid[goal[0], goal[1]] == 1:
            goal = self._find_nearest_navigable(goal)
            if goal is None:
                return None
        
        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1
        
        # Track visited nodes
        came_from = {}
        
        # Cost from start to node
        g_score = {start: 0}
        
        # Estimated total cost
        f_score = {start: self.heuristic(start, goal)}
        
        # Set of visited nodes
        closed_set = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            # Goal reached
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                # Diagonal movement costs sqrt(2), orthogonal costs 1
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if (dx + dy) == 2 else 1.0
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path is better
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
        
        # No path found
        return None
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _find_nearest_navigable(self, pos: Tuple[int, int], max_distance: int = 50) -> Optional[Tuple[int, int]]:
        """Find nearest navigable cell within max_distance"""
        x, y = pos
        
        for distance in range(1, max_distance):
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.height and 0 <= ny < self.width:
                        if self.grid[nx, ny] == 0:
                            return (nx, ny)
        
        return None


def compute_path(image, obstacle_grid, start_pos=None, goal_pos=None):
    """
    Compute optimal path and overlay on image
    
    Args:
        image: Original BGR image
        obstacle_grid: Binary obstacle map (1=obstacle, 0=navigable)
        start_pos: Optional start position (col, row). Default: bottom center
        goal_pos: Optional goal position (col, row). Default: top center
        
    Returns:
        path_image: Image with path overlay
        path: List of path coordinates
    """
    height, width = image.shape[:2]
    
    # Default start and goal
    if start_pos is None:
        start_pos = (width // 2, height - 20)
    if goal_pos is None:
        goal_pos = (width // 2, 20)
    
    # Convert from (col, row) to (row, col) for grid indexing
    start = (start_pos[1], start_pos[0])
    goal = (goal_pos[1], goal_pos[0])
    
    # Initialize path planner
    planner = AStarPathPlanner(obstacle_grid)
    
    # Find path
    path = planner.find_path(start, goal)
    
    # Create visualization
    path_image = image.copy()
    
    if path is not None and len(path) > 1:
        # Draw path
        path_points = [(p[1], p[0]) for p in path]  # Convert back to (col, row)
        
        # Draw path line
        for i in range(len(path_points) - 1):
            cv2.line(path_image, path_points[i], path_points[i + 1], (0, 255, 0), 4)
        
        # Draw start marker
        cv2.circle(path_image, start_pos, 10, (255, 0, 0), -1)  # Blue circle
        cv2.putText(path_image, "START", (start_pos[0] - 30, start_pos[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw goal marker
        cv2.circle(path_image, goal_pos, 10, (0, 0, 255), -1)  # Red circle
        cv2.putText(path_image, "GOAL", (goal_pos[0] - 25, goal_pos[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add path length info
        path_length = len(path)
        cv2.putText(path_image, f"Path Length: {path_length} pixels",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # No path found
        cv2.putText(path_image, "NO PATH FOUND", (width // 2 - 100, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return path_image, path