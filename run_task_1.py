from typing import List, Optional
import numpy as np
from functools import lru_cache

CELL_RESOLUTION = 1
CELL_HALF_SIZE = CELL_RESOLUTION / 2

INT_32_MAX = np.iinfo(np.int32).max

def angle_normalized_2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Get the angle between two vectors a and b in degrees.
    """
    delta = a - b
    cell_angle_rad = np.atan2(delta[1], delta[0])
    return np.degrees(cell_angle_rad)

@lru_cache(maxsize=None)
def get_meshgrid(grid_shape: tuple[int, int]) -> np.ndarray:
    x_vertices =  np.arange(0, grid_shape[1], CELL_RESOLUTION, dtype=np.int32)
    y_vertices = -np.arange(0, grid_shape[0], CELL_RESOLUTION, dtype=np.int32)
    xx_vertices, yy_vertices = np.meshgrid(x_vertices, y_vertices)
    vertices = np.hstack((xx_vertices.reshape(-1, 1), yy_vertices.reshape(-1, 1)))
    return vertices

class Cop:
    def __init__(self, id: int, index: tuple[int, int], orientation: int, fov: int, grid_shape: tuple[int, int]):
        self.id = id
        self.index = index
        self.orientation = orientation
        self.fov = fov
        self.grid_shape = grid_shape

        self.thief_pos: Optional[tuple[int, int]] = None
        self.visibility = np.zeros(grid_shape, dtype=bool)

        # This is in world coordinates, assuming the grid is in the fourth quadrant
        self.position = np.array([[CELL_HALF_SIZE+index[1]], [-CELL_HALF_SIZE-index[0]]], dtype=np.float32)

        self.compute_visibility()

    def compute_visibility(self, ):
        vertices = get_meshgrid(self.grid_shape)
        # Note that this angle is between zero to 360 degrees
        vertices_angle_T_cop = angle_normalized_2(vertices.T, self.position) - self.orientation
        vertices_visible_x, vertices_visible_y = np.where(np.abs(vertices_angle_T_cop.reshape(self.grid_shape)) <= self.fov/2)

        cell_indices_x = np.hstack((vertices_visible_x, vertices_visible_x-1, vertices_visible_x))
        cell_indices_y = np.hstack((vertices_visible_y, vertices_visible_y, vertices_visible_y-1))
        cell_indices = np.vstack((cell_indices_x, cell_indices_y)).T
        cell_indices_relavent_mask = np.where((cell_indices[:, 0] >= 0) & 
                                              (cell_indices[:, 0] < self.grid_shape[0]) & 
                                              (cell_indices[:, 1] >= 0) & 
                                              (cell_indices[:, 1] < self.grid_shape[1]))
        cell_indices_relavent = cell_indices[cell_indices_relavent_mask]

        self.visibility[cell_indices_relavent[:, 0], cell_indices_relavent[:, 1]] = True

    def is_cell_visible(self, cell_index: tuple[int, int]) -> bool:
        return self.visibility[cell_index[0], cell_index[1]]

def thief_and_cops(grid: List[List[str]], orientations, fov):
    """
    Assumptions:
    - There is only one thief in the grid and is represented by 'T'.
    - The cops are represented by numbers starting from 1.
    - The the empty spaces are represented by '0'.
    """
    assert len(orientations) == len(fov), "Number of orientations must match number of FOV angles"

    # Get the cop positions
    grid_shape = (len(grid), len(grid[0]))
    total_cop_vision = np.zeros((len(grid), len(grid[0])), dtype=bool)
    thief_pos = None
    cops: List[Cop] = []

    for grid_row_index, grid_row in enumerate(grid):
        for grid_col_index, cell in enumerate(grid_row):
            if cell.isdigit() and int(cell) > 0:
                cops.append(Cop(int(cell),
                                (grid_row_index, grid_col_index), 
                                orientations[int(cell)-1], 
                                fov[int(cell)-1],
                                grid_shape))

            if cell == 'T':
                thief_pos = (grid_row_index, grid_col_index)

    # Combine the cop visions
    cops_that_can_see_thief = []
    for cop in cops:
        total_cop_vision = np.logical_or(total_cop_vision, cop.visibility)
        if cop.is_cell_visible(thief_pos):
            cops_that_can_see_thief.append(cop.id)
    
    grid_row_index = np.arange(0, grid_shape[0], CELL_RESOLUTION, dtype=np.int32)
    grid_col_index = np.arange(0, grid_shape[1], CELL_RESOLUTION, dtype=np.int32)
    dist_from_thief_x = grid_row_index - thief_pos[0]
    dist_from_thief_y = grid_col_index - thief_pos[1]

    xx_dist_from_thief, yy_dist_from_thief = np.meshgrid(dist_from_thief_y, dist_from_thief_x)
    dist_from_thief = (np.abs(xx_dist_from_thief) + np.abs(yy_dist_from_thief))
    dist_from_thief[total_cop_vision] = INT_32_MAX

    # Get the nearest cell to the thief
    best_cell_flattened = dist_from_thief.argmin().item()
    best_cell = (best_cell_flattened // len(grid[0]), best_cell_flattened % len(grid[0]))
    if dist_from_thief[best_cell] == INT_32_MAX:
        best_cell = None

    return cops_that_can_see_thief, best_cell

if __name__ == '__main__':
    # Default Test Case
    grid = [['0', '0', '0', '0', '0'],
            ['T', '0', '0', '0', '2'], 
            ['0', '0', '0', '0', '0'], 
            ['0', '0', '1', '0', '0'], 
            ['0', '0', '0', '0', '0']]
    orientations = [180, 150]
    fov = [60, 60]
    result = thief_and_cops(grid, orientations, fov)
    assert result == ([2], (2, 2)), f"Expected ([1], (0, 2)), but got {result}"
    print(f"Default Test Case: {result}")

    # Test Case 1: Single Row, One Cop, One Thief
    grid_1 = [
        ['0', '0', 'T', '0', '1']
    ]
    orientations_1 = [0]
    fov_1 = [90]
    result_1 = thief_and_cops(grid_1, orientations_1, fov_1)
    assert result_1 == ([], (0, 2)), f"Expected ([1], (0, 2)), but got {result_1}"
    print("Test Case 1:", result_1)

    # Test Case 2: Single Row, One Cop, One Thief
    grid_2 = [
        ['0', '0', 'T', '0', '1']
    ]
    orientations_2 = [180]
    fov_2 = [90]
    result_2 = thief_and_cops(grid_2, orientations_2, fov_2)
    print("Test Case 2:", result_2)

    # Test Case 3: Single Column, One Cop, One Thief
    grid_3 = [['0'],
              ['0'], 
              ['T'], 
              ['0'], 
              ['1']] 
    orientations_3 = [270]
    fov_3 = [90]
    result_3 = thief_and_cops(grid_3, orientations_3, fov_3)
    print("Test Case 3:", result_3)

    # Test Case 4: Single Column, One Cop, One Thief
    grid_4 = [['0'],
              ['0'], 
              ['T'], 
              ['0'], 
              ['1']] 
    orientations_4 = [90]
    fov_4 = [90]
    result_4 = thief_and_cops(grid_4, orientations_4, fov_4)
    assert result_4 == ([1], None), f"Expected ([1], None), but got {result_4}"
    print("Test Case 4:", result_4)

    grid_5 = [
        ['0', '0', '0', '0', '0'],
        ['T', '0', '0', '0', '0'],
        ['0', '0', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '0', '0']
    ]
    orientations_5 = [0]
    fov_5 = [360]
    result_5 = thief_and_cops(grid_5, orientations_5, fov_5)
    assert result_5 == ([1], None), f"Expected ([1], None), but got {result_5}"
    print("Test Case 5:", result_5)

    grid_6 = [
        ['0', '0', '1', '2', '3'],
        ['T', '0', '0', '0', '0'],
        ['0', '0', '0', '0', '0']
    ]
    orientations_6 = [0, 90, 270]
    fov_6 = [45, 60, 120]
    result_6 = thief_and_cops(grid_6, orientations_6, fov_6)
    print("Test Case 6:", result_6)

