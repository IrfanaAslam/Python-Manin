from manim import *
import numpy as np

# --- Configuration ---
# Vectors for addition
# Manim uses numpy arrays for vectors internally, typically (x, y, z)
# For 2D, z-component is 0
VECTOR_A_NP = np.array([2.0, 3.0, 0.0])
VECTOR_B_NP = np.array([-1.0, 2.0, 0.0])

# Vector for scalar multiplication
VECTOR_V_NP = np.array([3.0, 1.0, 0.0])
SCALAR_MIN = -2.0
SCALAR_MAX = 3.0
ANIMATION_DURATION_SCALAR = 4.0 # Duration for scalar multiplication part

class VectorOperationsVisualization(Scene):
    def construct(self):
        # --- Setup the coordinate system ---
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            x_length=12,
            y_length=12,
            axis_config={"color": GRAY},
            tips=False
        ).add_coordinates()
        self.play(Create(axes), run_time=1)
        self.wait(0.5)

        # --- Vector Addition ---
        self.next_section("Vector Addition")
        title_add = Text("Vector Addition", font_size=40).to_edge(UP)
        self.play(Write(title_add))
        self.wait(0.5)

        # Vector A
        vec_a = Vector(VECTOR_A_NP, color=BLUE)
        label_a = MathTex("A").next_to(vec_a.get_end(), RIGHT + UP * 0.2)
        self.play(Create(vec_a), Write(label_a))
        self.wait(0.5)

        # Vector B (from origin)
        vec_b_origin = Vector(VECTOR_B_NP, color=GREEN)
        label_b_origin = MathTex("B").next_to(vec_b_origin.get_end(), LEFT + UP * 0.2)
        self.play(Create(vec_b_origin), Write(label_b_origin))
        self.wait(1)

        # Move vector B to the tip of A
        # Create a new vector B for the shifted version
        vec_b_shifted = Vector(VECTOR_B_NP).shift(vec_a.get_end())
        self.play(Transform(vec_b_origin, vec_b_shifted), run_time=1.5) # Animate transforming original B to shifted B
        self.wait(0.5)

        # Resultant vector A + B
        vec_sum_np = VECTOR_A_NP + VECTOR_B_NP
        vec_sum = Vector(vec_sum_np, color=RED)
        label_sum = MathTex("A + B").next_to(vec_sum.get_end(), UP + RIGHT * 0.5)

        self.play(Create(vec_sum), Write(label_sum))
        self.wait(1)

        # Show parallelogram rule (dashed lines)
        # Line from original tip of B to tip of A+B (parallel to A)
        para_a_line_corrected = DashedLine(VECTOR_B_NP, vec_sum_np, color=BLUE_A) # Use original VECTOR_B_NP
        # Line from tip of A to tip of A+B (parallel to B)
        para_b_line_corrected = DashedLine(VECTOR_A_NP, vec_sum_np, color=GREEN_A) # Use original VECTOR_A_NP

        self.play(Create(para_a_line_corrected), Create(para_b_line_corrected), run_time=1.5)
        self.wait(1)

        # Fade out vector addition elements
        self.play(
            FadeOut(title_add, vec_a, label_a, vec_b_origin, label_b_origin, vec_sum, label_sum,
                     para_a_line_corrected, para_b_line_corrected)
        )
        self.wait(1)

        # --- Scalar Multiplication ---
        self.next_section("Scalar Multiplication")
        title_scalar = Text("Scalar Multiplication", font_size=40).to_edge(UP)
        self.play(Write(title_scalar))
        self.wait(0.5)

        # Original Vector V
        vec_v = Vector(VECTOR_V_NP, color=PURPLE)
        label_v = MathTex("V").next_to(vec_v.get_end(), RIGHT * 0.5 + UP * 0.2)
        self.play(Create(vec_v), Write(label_v))
        self.wait(1)

        # Initialize ValueTracker for scalar value
        # This is the Mobject whose value will be animated
        scalar_tracker = ValueTracker(SCALAR_MIN) 

        # Animate scalar multiplication using ValueTracker
        scaled_vec = always_redraw(lambda: Vector(
            VECTOR_V_NP * scalar_tracker.get_value(), color=ORANGE
        ))
        scaled_label = always_redraw(lambda: MathTex(f"k V = {scalar_tracker.get_value():.1f} V").next_to(
            scaled_vec.get_end(), UP + RIGHT * 0.5
        ))
        
        self.add(scaled_vec, scaled_label) # Add to scene now

        self.play(
            scalar_tracker.animate.set_value(SCALAR_MAX), # Animate the ValueTracker
            run_time=ANIMATION_DURATION_SCALAR,
            rate_func=linear
        )
        self.wait(0.5)

        # Animate scalar value going back to initial
        self.play(
            scalar_tracker.animate.set_value(SCALAR_MIN), # Animate the ValueTracker
            run_time=ANIMATION_DURATION_SCALAR,
            rate_func=linear
        )
        self.wait(1)

        # Fade out scalar multiplication elements
        self.play(
            FadeOut(title_scalar, vec_v, label_v, scaled_vec, scaled_label)
        )
        self.wait(0.5)

        self.play(FadeOut(axes))
        self.wait(1)

    # The custom scalar_updater method is no longer needed with ValueTracker
    # def scalar_updater(self, target_value):
    #     """Creates an animation for scalar_value attribute."""
    #     def updater(mobject, alpha):
    #         self.scalar_value = interpolate(self.scalar_value_start, target_value, alpha)
        
    #     # Ensure scalar_value_start is initialized for the first call
    #     if not hasattr(self, 'scalar_value_start'):
    #         # This check is less critical now that self.scalar_value is initialized earlier
    #         self.scalar_value_start = self.scalar_value if hasattr(self, 'scalar_value') else 0.0
        
    #     return UpdateFromAlphaFunc(self, updater)
