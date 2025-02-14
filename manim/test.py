from manim import *
import numpy as np

class MovingPointOnSurface(ThreeDScene):
    def construct(self):
        # Set up the 3D camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Define the surface function
        surface = Surface(
            lambda u, v: np.array([u, v, np.sin(u) * np.cos(v)]),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(50, 50),
        )

        # Apply gradient color based on height (z-value)
        #surface.set_color_by_gradient(BLUE, GREEN, YELLOW, lambda x, y, z: (z + 1) / 2)
        # Create a moving dot
        dot = Dot3D(color=RED, radius=0.08)

        # Function to update dot position
        def update_dot(mob, alpha):
            u = interpolate(-3, 3, alpha)  # Move u from -3 to 3
            v = u  # Define v as a function of u (optional movement)
            x, y, z = np.array([u, v, np.sin(u) * np.cos(v)])  # Compute surface position
            mob.move_to([x, y, z])  # Update dot position

        # Animate the dot moving along the surface
        self.add(surface, dot)
        self.play(UpdateFromAlphaFunc(dot, update_dot), run_time=5, rate_func=linear)

        # Hold the final position
        self.wait(2)
