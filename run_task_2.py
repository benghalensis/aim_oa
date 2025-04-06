import argparse
import os
import cv2
import yaml
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import math

import numpy as np
import torch
from sklearn.cluster import KMeans

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from demo.predictor import VisualizationDemo

np.set_printoptions(suppress=True)

def setup_cfg(config_file: str, opts: List[str], confidence_threshold: float) -> get_cfg:
    """
    Configure the Detectron2 model with the provided parameters.
    
    Args:
        config_file: Path to the model configuration file.
        opts: List of options to modify the configuration.
        confidence_threshold: Threshold for prediction confidence.
        
    Returns:
        The configured Detectron2 configuration object.
    """
    # Load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

class ObjectTracker:
    """
    Particle filter-based object tracker for tracking balls in video sequences.
    
    The tracker uses a combination of Detectron2 object detection and 
    particle filtering to robustly track balls across frames.
    
    Attributes:
        max_num_particles: Maximum number of particles in the filter.
        num_particles: Number of particles excluding random particles.
        frame_width: Width of the video frame.
        frame_height: Height of the video frame.
        state: Current state of particles [x, y, radius, vx, vy, vr].
        ball_probabilities: Probability of each particle being the ball.
    """
    def __init__(self, config: Dict[str, Any], device, debug=False):
        """
        Initialize the object tracker with the provided configuration.
        
        Args:
            config: Dictionary containing tracker configuration parameters.
        """
        # Particle filter parameters
        self.max_num_particles = int(config['num_particles'])
        random_ratio = config['random_particles_ratio']
        self.num_particles = int(self.max_num_particles * (1 - random_ratio))
        self.device = device
        self.measurement_noise = torch.tensor(config['measurement_noise'], device=self.device, dtype=torch.float32)
        self.motion_noise = torch.tensor(config['motion_noise'], device=self.device, dtype=torch.float32)

        # Frame dimensions
        self.frame_width = config['frame_width']
        self.frame_height = config['frame_height']
        self.init_guess_ball_radius = config['init_guess_ball_radius']
        
        # Scale for random particle generation
        self.random_particles_scale = torch.tensor([
            self.frame_width,
            self.frame_height, 
            self.init_guess_ball_radius, 
            0, 0, 0
        ], dtype=torch.float32, device=self.device)

        # Initialize particles based on distribution parameters
        dist_min = torch.tensor(config['init_distribution_min'], device=self.device, dtype=torch.float32)
        dist_max = torch.tensor(config['init_distribution_max'], device=self.device, dtype=torch.float32)
        self.state = torch.rand(self.num_particles, 6, device=self.device, dtype=torch.float32) * (dist_max - dist_min) + dist_min

        # Initialize probabilities with small values
        self.ball_probabilities = torch.ones(self.num_particles, device=self.device, dtype=torch.float32) * 0.0001

        # Fill with random particles
        self.fill_up_with_random_particles(self.max_num_particles - self.num_particles)

        self.final_state = torch.zeros(1, 6, device=self.device, dtype=torch.float32)
        self.dt = 1
        self.debug = debug

        if debug:
            self.debug_dir = "data/debug"
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs("data/dog", exist_ok=True)

    def update(self, frame, count, boxes, pred_classes, pred_scores):
        """
        Update the tracker with the current frame and predictions.
        
        Args:
            frame: The current video frame.
            count: Frame count.
            boxes: Bounding boxes of detected objects.
            pred_classes: Predicted classes of detected objects.
            pred_scores: Prediction scores of detected objects.
        """
        # Predict to the next frame
        self.predict(frame)

        # Correct based on the predictions
        self.correct(frame, count, boxes, pred_classes, pred_scores)

        # Visualize the predictions
        self.visualize(frame, count) if self.debug else None

    def predict(self, frame):
        """
        Predict the next state of the particles based on a noisy motion model. This uses a constant velocity model.
        """
        # Generate noise
        motion_noise = torch.randn_like(self.state) * self.motion_noise
        self.state = self.state + motion_noise

        new_state = torch.zeros_like(self.state, device=self.device, dtype=torch.float32)
        new_state = torch.cat((self.state[:, 0:3] + self.state[:, 3:6] * self.dt, self.state[:, 3:6]), dim=1)

        # Update the final state without noise
        self.final_state = torch.cat((self.final_state[:, 0:3] + self.final_state[:, 3:6] * self.dt, self.final_state[:, 3:6]), dim=1)

        # Remove the particles that are outside the frame
        in_frame_mask = self.in_frame_mask(new_state[:,0], new_state[:,1], frame.shape)
        self.state = new_state[in_frame_mask]
        if self.ball_probabilities is not None:
            self.ball_probabilities = self.ball_probabilities[in_frame_mask]
        
        self.fill_up_with_random_particles(self.max_num_particles - self.ball_probabilities.shape[0])

    def correct(self, frame, count, boxes, pred_classes, pred_scores):
        """
        Correct the particle states based on the observed predictions.
        
        Args:
            frame: The current video frame.
            count: Frame count.
            boxes: Bounding boxes of detected objects.
            pred_classes: Predicted classes of detected objects.
            pred_scores: Prediction scores of detected objects.
        """
        observed_ball_probabilities = self.prediction_probability(frame, count, boxes, pred_classes, pred_scores)

        if observed_ball_probabilities is None:
            # If there is no deep learning prediction, then use the DoG blob finder
            observed_ball_probabilities = self.dof_generator(frame, count)
            return

        # Get the top particles that were recently observed
        ball_probabilities = observed_ball_probabilities / torch.max(observed_ball_probabilities)
        states = self.state

        # Get the indices of the top 10 values in ball_probabilities
        topk = torch.topk(ball_probabilities, 10)
        clustering_states = states[topk.indices]
        self.final_state = clustering_states.mean(dim=0, keepdim=True)

        normalized_ball_probabilities = ball_probabilities / ball_probabilities.sum()
        resampled_indices = torch.multinomial(normalized_ball_probabilities, self.num_particles, replacement=True)
        self.state = states[resampled_indices]
        self.ball_probabilities = ball_probabilities[resampled_indices]

        # Generate random particles
        self.fill_up_with_random_particles(self.max_num_particles - self.num_particles)
        pass

    def fill_up_with_random_particles(self, num_random_particles): 
        """
        Fill the state with random particles.
        
        Args:
            num_random_particles: Number of random particles to generate.
        """
        random_particle_states = torch.rand(num_random_particles, 6, device=self.device, dtype=torch.float32) * self.random_particles_scale
        random_ball_probabilities = torch.ones(num_random_particles, device=self.device, dtype=torch.float32) * 0.0001
        self.state = torch.cat((self.state, random_particle_states), dim=0)
        self.ball_probabilities = torch.cat((self.ball_probabilities, random_ball_probabilities), dim=0)

    def find_largest_cluster(self, states):
        """
        Find the largest cluster of particles using KMeans clustering.
        
        Args:
            states: The current state of particles.
            
        Returns:
            The mean state of the largest cluster.
        """
        kmeans = KMeans(n_clusters=3, random_state=0).fit(states[:, 0:2])
        labels = kmeans.labels_
        largest_cluster_label = max(set(labels), key=list(labels).count)
        largest_cluster_states = states[labels == largest_cluster_label]
        return np.mean(largest_cluster_states, axis=0)[None]

    def prediction_probability(self, frame, count, boxes, pred_classes, pred_scores):
        """
        Calculate the probability of each particle being the ball based on predictions.
        
        Args:
            frame: The current video frame.
            count: Frame count.
            boxes: Bounding boxes of detected objects.
            pred_classes: Predicted classes of detected objects.
            pred_scores: Prediction scores of detected objects.
            
        Returns:
            The probability of each particle being the ball.
        """
        # Get all the predictions of importance
        sports_ball_mask = (pred_classes == 32) # [N, ]
        if sports_ball_mask.sum() == 0:
            return None
        sports_ball_boxes = boxes[sports_ball_mask] # [N, 4]
        sports_ball_center = (sports_ball_boxes[:, 0:2] + sports_ball_boxes[:, 2:4]) / 2 # [N, 2]
        sports_ball_radius = torch.mean(sports_ball_boxes[:, 2:4] - sports_ball_boxes[:, 0:2], dim=1, keepdim=True) / 2 # [N, 1]

        sports_ball_state = torch.cat((sports_ball_center, sports_ball_radius), dim=1)
        sports_ball_state = sports_ball_state[sports_ball_state[:, 2] > 10]

        if sports_ball_state.numel() == 0:
            return None
        observed_pos = sports_ball_state.detach().T

        g1 = torch.exp(-torch.square(self.state[:, 0:3, None] - observed_pos[None, :, :]) / (2 * torch.square(self.measurement_noise[None, 0:3, None]))) / (math.sqrt(2 * math.pi) * self.measurement_noise[None, 0:3, None])
        g2 = torch.prod(g1, dim=1)
        g3 = torch.max(g2, dim=1).values
        return g3

    def dof_generator(self, frame, count):
        """
        Generate probabilities using Difference of Gaussian (DoG) blob detection.
        
        Args:
            frame: The current video frame.
            count: Frame count.
            
        Returns:
            The probability of each particle being the ball.
        """
        # Apply the gaussian blur to the frame
        frame = frame.detach().cpu().numpy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gaussian_blur_1 = cv2.GaussianBlur(frame_gray, (45, 45), 0)
        frame_gaussian_blur_2 = cv2.GaussianBlur(frame_gray, (65, 65), 0)
        dog = np.abs(frame_gaussian_blur_1.astype(np.float32) - frame_gaussian_blur_2.astype(np.float32))
        cv2.normalize(dog, dog, 0, 1.0, cv2.NORM_MINMAX)
        dog[dog < 0.5] = 0
        cv2.normalize(dog, dog, 0, 1.0, cv2.NORM_MINMAX)

        if self.debug:
            # Visualize the DoG
            dog_viz = dog.copy()
            dog_viz = (dog_viz * 255).astype(np.uint8)
            cv2.imwrite(f"data/dog/{count:04d}.png", dog_viz)
        
        dog = torch.tensor(dog, device=self.device, dtype=torch.float32)
        probablities = dog[self.state[:, 1].long(), self.state[:, 0].long()]
        return probablities

    def in_frame_mask(self, x_indices, y_indices, frame_shape):
        """
        Create a mask to filter particles within the frame boundaries.
        
        Args:
            x_indices: X coordinates of particles.
            y_indices: Y coordinates of particles.
            frame_shape: Shape of the video frame.
            
        Returns:
            A boolean mask indicating particles within the frame.
        """
        return (x_indices >= 0) & (x_indices < frame_shape[1]) & (y_indices >= 0) & (y_indices < frame_shape[0])

    def get_circumference_pixels(self, radius):
        """
        Get the pixel coordinates of the circumference of particles.
        
        Args:
            radius: Radius of particles.
            
        Returns:
            Pixel coordinates of the circumference.
        """
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        px = (self.state[:, 0, None] + radius[:, None] * np.cos(angles)[None, :]).astype(int)
        py = (self.state[:, 1, None] + radius[:, None] * np.sin(angles)[None, :]).astype(int)
        return np.stack([px, py], axis=0).astype(int)
    
    def get_radial_vector(self ):
        """
        Get the radial vector for particles.
        
        Returns:
            Radial vector for particles.
        """
        angles = np.linspace(0, 2*np.pi, 16, dtype=np.float32, endpoint=False)
        px = np.cos(angles)
        py = np.sin(angles)
        return np.stack([px, py], axis=0).T
    
    def visualize(self, frame, count):
        """
        Visualize the particles and final state on the frame.
        
        Args:
            frame: The current video frame.
            count: Frame count.
        """
        frame = frame.detach().cpu().numpy()
        particle_viz = self.visualize_particles(frame, count)
        final_state = self.final_state.detach().cpu().numpy()

        # Draw the final state
        cv2.circle(frame, (int(final_state[0,0]), int(final_state[0,1])), int(final_state[0,2]), (0, 255, 0), 3)
        viz_image = np.concatenate((particle_viz, frame), axis=0)
        cv2.imwrite(os.path.join(f"data/debug/frame_{count:04d}.png"), viz_image)

    def visualize_particles(self, frame, count):
        """
        Visualize the particles on the frame.
        
        Args:
            frame: The current video frame.
            count: Frame count.
            
        Returns:
            The visualized particles.
        """
        # This will be using a particle filter therefore add a bunch of red dots for the particles
        state = self.state.detach().cpu().numpy()
        ball_probabilities = self.ball_probabilities.detach().cpu().numpy()

        writable_frame = np.zeros(frame.shape[0:2], dtype=np.uint8)
        coords = (state[:,1].astype(int), state[:,0].astype(int))
        prob_values = (ball_probabilities / ball_probabilities.max() * 255).astype(np.uint8)
        np.maximum.at(writable_frame, coords, prob_values)
        return cv2.applyColorMap(writable_frame, cv2.COLORMAP_JET)
    
    def get_state_info(self):
        """
        Get the final state of the tracker.
        
        Returns:
            {x_center}, {y_center}, {x_size}, {y_size}
        """
        return [self.final_state[0, 0].item(), self.final_state[0, 1].item(), 2*self.final_state[0, 2].item(), 2*self.final_state[0, 2].item()]


def draw_bounding_box(frame, state_info):
    """
    Draw a bounding box around the tracked object.
    
    Args:
        frame: The current video frame.
        state_info: Information about the tracked object.
        
    Returns:
        The frame with the bounding box drawn.
    """
    x_center, y_center, x_size, y_size = state_info
    x1 = int(x_center - x_size / 2)
    y1 = int(y_center - y_size / 2)
    x2 = int(x_center + x_size / 2)
    y2 = int(y_center + y_size / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return frame

def main(input_dir, output_dir, config_file):
    """
    Main function to run the object tracking.
    
    Args:
        input_dir: Directory containing input frames.
        output_dir: Directory to save output frames.
        config_file: Path to the configuration file.
    """
    # Load the yaml config file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Create the output directory if it doesn't exist
    os.makedirs(f"data/results", exist_ok=True)

    # Check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup the detectron2 config
    cfg = setup_cfg(config['detectron2']['config_file'], config['detectron2']['opts'], config['detectron2']['confidence_threshold'])
    demo = VisualizationDemo(cfg)

    # Setup the object tracker
    tracker_config = {
        'frame_width': 640,
        'frame_height': 360,
        'init_guess_ball_radius': 20,
        'random_particles_ratio': 0.2,
        'num_particles': 10000,
        'init_distribution_min': [450, 285, 15, -1, -1, -0.05],
        'init_distribution_max': [485, 300, 30, 1, 1, 0.05],
        'motion_noise': [0, 0, 0, 3, 3, 0.1],
        'measurement_noise': [30, 30, 5],
    }
    tracker = ObjectTracker(tracker_config, device, debug=True)

    # Get frames from the inuput directory and sort them
    frames = sorted([i for i in os.listdir(input_dir) if i.endswith('.png')])

    # Open a csv file to save the results
    file = open("results.csv", "w")

    for count, frame in tqdm(enumerate(frames)):
        # Read the image
        img = read_image(os.path.join(input_dir, frame), format="BGR")
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

        predictions, visualized_output = demo.run_on_image(img)
        boxes = predictions["instances"].pred_boxes.tensor
        pred_classes = predictions["instances"].pred_classes
        pred_scores = predictions["instances"].scores

        # Convert the image to numpy array
        img_tensor = torch.tensor(img, requires_grad=False).to(device=device)
        tracker.update(img_tensor, count, boxes, pred_classes, pred_scores)

        tracker_state = tracker.get_state_info()
        file.write(f"{count},{tracker_state[0]},{tracker_state[1]},{tracker_state[2]},{tracker_state[3]}\n")

        result_viz = draw_bounding_box(img, tracker_state)
        cv2.imwrite(f"data/results/{count:04d}.png", result_viz)
    
    file.close()
    print("Tracking completed. Results saved to results.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Tracking of ball")
    parser.add_argument("--input_dir", type=str, default="data/frames", help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, default="data/output", help="Directory to save output files") 
    parser.add_argument("--config_file", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.config_file)