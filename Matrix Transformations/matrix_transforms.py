from manim import *

class MatrixTransformations(Scene):
    def construct(self):
        # --- Configuration ---
        # Manim's Square is simpler for this specific case
        original_square = Square(side_length=1, color=WHITE).move_to(ORIGIN + 0.5 * RIGHT + 0.5 * UP)
        
        # Transformation matrices to demonstrate
        SCALE_MATRIX = np.array([[1.5, 0], [0, 0.5]])
        THETA = np.radians(45)
        ROTATION_MATRIX = np.array([[np.cos(THETA), -np.sin(THETA)],
                                    [np.sin(THETA), np.cos(THETA)]])
        SHEAR_MATRIX = np.array([[1, 0.8], [0, 1]])

        # Extend 2x2 matrices to 3x3 for Manim's ApplyMatrix
        def extend_matrix(mat2x2):
            mat3x3 = np.identity(3)
            mat3x3[:2, :2] = mat2x2
            return mat3x3

        TRANSFORMATIONS = [
            ("Scaling", extend_matrix(SCALE_MATRIX), BLUE),
            ("Rotation (45°)", extend_matrix(ROTATION_MATRIX), GREEN),
            ("Shear (X-axis)", extend_matrix(SHEAR_MATRIX), PURPLE)
        ]

        # --- Scene Setup ---
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            background_line_style={
                "stroke_color": GREY,
                "stroke_opacity": 0.6,
            }
        ).add_coordinates()
        
        plane.set_opacity(0.7)

        self.add(plane)
        self.play(Create(plane, run_time=1.5))

        # Label for the current transformation
        transform_label = Text("Original Shape", font_size=24, color=WHITE).to_edge(UP).shift(LEFT*3)
        self.add(transform_label)
        
        # Original square reference (static)
        # It's better to name it clearly as a reference or background square
        static_original_square = original_square.copy()
        static_original_square.set_stroke(color=WHITE, width=3).set_opacity(0.5) 
        self.add(static_original_square) 

        # The square that will be transformed
        transforming_square = original_square.copy()
        transforming_square.set_stroke(color=RED, width=4).set_color(RED)
        self.add(transforming_square)

        self.play(Create(transforming_square))
        self.wait(1)

        # Transformation loop
        for i, (name, matrix, color) in enumerate(TRANSFORMATIONS):
            # Update transformation name label
            new_label = Text(f"{name}", font_size=36, color=color).to_edge(UP)
            self.play(Transform(transform_label, new_label))
            
            # Apply the matrix transformation to the transforming_square
            self.play(
                ApplyMatrix(matrix, transforming_square, run_time=2, rate_func=smooth),
                transforming_square.animate.set_color(color), # Animate color change
            )
            self.wait(1.5) # Hold the transformed shape

            # Reset the transforming_square for the next transformation
            if i < len(TRANSFORMATIONS) - 1: # Don't reset after the last transformation
                # Create a temporary square that represents the original state with red color
                temp_reset_square = original_square.copy().set_color(RED).set_stroke(color=RED, width=4)
                
                # Transform the currently transformed square back to the original red square
                self.play(
                    Transform(transforming_square, temp_reset_square, run_time=1.5, rate_func=smooth),
                    # We don't need to animate static_original_square's opacity here repeatedly
                    # It stays at 0.5 throughout, providing a consistent reference.
                )
                self.wait(0.5)

        self.wait(2) # Final hold