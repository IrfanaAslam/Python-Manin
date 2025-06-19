from manim import *

class PolynomialLongDivision(Scene):
    def construct(self):
        # --- 1. Define Polynomials ---
        # Represent polynomials as lists of (coefficient, exponent) tuples.
        # This makes arithmetic easier than string manipulation for calculations.
        # Example: x^3 - 2x^2 - 4x + 8
        self.dividend_coeffs = [(1, 3), (-2, 2), (-4, 1), (8, 0)]
        # Example: x - 2
        self.divisor_coeffs = [(1, 1), (-2, 0)]

        # Determine the variable to use in the polynomials
        self.var = 'x'

        # Store calculated quotient terms directly as (coeff, exp) tuples
        self.quotient_terms_calculated = []

        # --- 2. Initial Setup: Display Dividend, Divisor, and Division Bracket ---
        self.setup_division_display(self.dividend_coeffs, self.divisor_coeffs)

        # --- 3. Perform Long Division Steps ---
        self.perform_long_division(self.dividend_coeffs, self.divisor_coeffs)

        # --- 4. Final Result Display ---
        self.display_final_result()

    # --- Helper Functions for Polynomial Arithmetic and Manim Display ---

    def poly_to_tex_str(self, poly_coeffs, var=None):
        """Converts a list of (coefficient, exponent) tuples to a LaTeX string."""
        if var is None:
            var = self.var

        if not poly_coeffs:
            return "0"

        # Sort terms by exponent in descending order
        sorted_coeffs = sorted(poly_coeffs, key=lambda x: x[1], reverse=True)

        terms_str = []
        for coeff, exp in sorted_coeffs:
            if coeff == 0:
                continue # Skip zero terms

            term_part = ""
            # Handle sign for positive coefficients after the first term, or for any negative coefficient
            if coeff > 0 and len(terms_str) > 0:
                term_part += "+"
            elif coeff < 0:
                term_part += "-"

            abs_coeff = abs(coeff)
            # Handle coefficient: don't show '1' for x^1 or x^N, but show for constants like '1'
            if abs_coeff != 1 or exp == 0:
                term_part += str(int(abs_coeff)) # Cast to int to avoid .0 for whole numbers

            # Handle variable and exponent
            if exp > 0:
                term_part += var
                if exp > 1:
                    term_part += "^{" + str(exp) + "}"
            terms_str.append(term_part)

        return "".join(terms_str) if terms_str else "0"

    def get_tex_mobjects_for_poly(self, poly_coeffs, var=None):
        """
        Creates a LIST of individual MathTex objects for each term in a polynomial.
        """
        if var is None:
            var = self.var

        if not poly_coeffs:
            return [MathTex("0")] # Return a list containing '0' MathTex

        mobjects = []
        for coeff, exp in poly_coeffs:
            if coeff == 0:
                continue

            # This part generates the string for a single term, including its sign
            term_str = self.poly_to_tex_str([(coeff, exp)], var=var)
            mobjects.append(MathTex(term_str))

        return mobjects # Return list of MathTex objects


    def normalize_poly(self, poly_coeffs):
        """Removes zero-coefficient terms and sorts by exponent."""
        normalized = []
        for coeff, exp in poly_coeffs:
            if abs(coeff) > 1e-9: # Consider very small numbers as zero due to float arithmetic
                normalized.append((coeff, exp))
        return sorted(normalized, key=lambda x: x[1], reverse=True)

    def get_leading_term(self, poly_coeffs):
        """Returns the leading (highest exponent) term (coeff, exp)."""
        normalized = self.normalize_poly(poly_coeffs)
        if not normalized:
            return (0, -1) # Represents zero polynomial or empty
        return normalized[0]

    def poly_degree(self, poly_coeffs):
        """Returns the highest exponent in the polynomial."""
        if not poly_coeffs:
            return -1 # Represents zero polynomial
        return self.get_leading_term(poly_coeffs)[1]

    def poly_term_divide(self, term1, term2):
        """Divides two (coeff, exp) terms. Returns (coeff, exp) of quotient."""
        coeff1, exp1 = term1
        coeff2, exp2 = term2
        if coeff2 == 0:
            raise ValueError("Division by zero term")
        if exp1 < exp2:
            return (0, -1) # Resulting term would have negative exponent
        return (coeff1 / coeff2, exp1 - exp2)

    def poly_multiply_term_by_poly(self, term, poly_coeffs):
        """Multiplies a single (coeff, exp) term by a polynomial (list of terms)."""
        product = []
        term_coeff, term_exp = term
        for poly_coeff, poly_exp in poly_coeffs:
            product.append((term_coeff * poly_coeff, term_exp + poly_exp))
        return self.normalize_poly(product)

    def poly_subtract(self, minuend_coeffs, subtrahend_coeffs):
        """
        Subtracts subtrahend from minuend.
        Combines terms with same exponent.
        """
        result_map = {}
        for coeff, exp in minuend_coeffs:
            result_map[exp] = result_map.get(exp, 0) + coeff
        for coeff, exp in subtrahend_coeffs:
            result_map[exp] = result_map.get(exp, 0) - coeff # Subtract

        result = [(coeff, exp) for exp, coeff in result_map.items() if abs(coeff) > 1e-9]
        return self.normalize_poly(result)


    def setup_division_display(self, dividend_coeffs, divisor_coeffs):
        """Sets up the initial display of the long division problem."""
        self.current_y_pos = 2.5 # Starting Y-position for the top row

        # Determine the range of exponents for consistent alignment
        max_exp_dividend = self.poly_degree(dividend_coeffs)
        
        self.column_width = 1.5 # Fixed horizontal spacing between terms/columns
        
        # Calculate ideal starting x based on where the highest exponent term of the dividend will be.
        initial_first_term_x = 1.0 # Approximate center X for the leading term of dividend

        self.exponent_x_map = {}
        for exp in range(max_exp_dividend, -1, -1): # Iterate from highest exp down to 0
            # Calculate x-position for this exponent
            self.exponent_x_map[exp] = initial_first_term_x - (max_exp_dividend - exp) * self.column_width

        # Display Divisor (left of bracket)
        # Use VGroup(*...) because get_tex_mobjects_for_poly now returns a list
        self.divisor_mobjects = VGroup(*self.get_tex_mobjects_for_poly(divisor_coeffs)).move_to(LEFT * 4 + UP * self.current_y_pos)
        self.play(Write(self.divisor_mobjects), run_time=1)

        # Display Dividend (inside bracket) - now positioned using the map
        all_dividend_term_mobjs = self.get_tex_mobjects_for_poly(dividend_coeffs)
        self.dividend_mobjects = VGroup()
        
        # Create a map from exponent to MathTex object for easier lookup and consistent ordering
        temp_dividend_map = {self.parse_tex_exponent(mobj.get_tex_string()): mobj for mobj in all_dividend_term_mobjs}
        
        # Iterate through expected exponents in descending order to build the VGroup
        for exp in sorted(self.exponent_x_map.keys(), reverse=True):
            if exp in temp_dividend_map:
                mobj = temp_dividend_map[exp]
                mobj.move_to(RIGHT * self.exponent_x_map[exp] + UP * self.current_y_pos)
                self.dividend_mobjects.add(mobj)
        
        self.play(Write(self.dividend_mobjects), run_time=1)

        # Draw Division Bracket (L-shaped line)
        self.bracket_length = self.dividend_mobjects.width + 1.0 # Extend slightly beyond dividend
        self.division_bracket = VGroup(
            Line(UP * 0.5, DOWN * 0.5)
                .set_height(self.dividend_mobjects.height * 1.5)
                .next_to(self.dividend_mobjects, LEFT, buff=0.2),
            Line(LEFT, RIGHT)
                .set_width(self.bracket_length)
                .next_to(self.dividend_mobjects, UP, buff=0.1)
                .align_to(self.dividend_mobjects, LEFT) # Align top bar with leftmost dividend term
        )
        self.play(Create(self.division_bracket), run_time=1)
        self.wait(1)

        # Initialize current_remainder_mobjects with the dividend for the first step
        self.current_remainder_mobjects = self.dividend_mobjects.copy()
        # Keep a reference to the initial full dividend for "dropping down" terms later
        self._original_dividend_full_terms = sorted(list(dividend_coeffs), key=lambda x: x[1], reverse=True)
        
        # Track exponents from the original dividend that have already been 'used' (processed in active coeffs or explicitly dropped)
        self.used_original_dividend_exponents = set()

        # Initialize quotient_mobjects here
        self.quotient_mobjects = VGroup()


    def perform_long_division(self, dividend_coeffs, divisor_coeffs):
        """Executes the main long division loop with animations."""
        # This holds the current mathematical polynomial segment being worked on.
        current_active_coeffs = list(dividend_coeffs)

        # The y-position for the terms below the current remainder
        self.current_y_pos_for_new_row = self.current_y_pos - self.dividend_mobjects.height * 1.5

        while self.poly_degree(current_active_coeffs) >= self.poly_degree(divisor_coeffs) and current_active_coeffs:
            # --- Get Leading Term for Division ---
            leading_active_term = self.get_leading_term(current_active_coeffs)
            leading_divisor_term = self.get_leading_term(divisor_coeffs)

            # --- Animate Highlighting of Leading Terms ---
            leading_active_mobj = None
            for mobj in self.current_remainder_mobjects:
                # Need to be careful with MathTex string comparison due to potential internal formatting
                term_str_to_match = self.poly_to_tex_str([leading_active_term]) # No replace here, MathTex handles it
                if mobj.get_tex_string() == term_str_to_match: # Exact match needed for single term
                    leading_active_mobj = mobj
                    break
            
            if leading_active_mobj:
                self.play(Indicate(leading_active_mobj), Indicate(self.divisor_mobjects[0]), run_time=1)
            elif self.current_remainder_mobjects: # Fallback: indicate first visible term if specific not found
                self.play(Indicate(self.current_remainder_mobjects[0]), Indicate(self.divisor_mobjects[0]), run_time=1)
            self.wait(0.2) # Small pause after indication

            # --- Calculate Quotient Term ---
            quotient_term_result = self.poly_term_divide(leading_active_term, leading_divisor_term)
            self.quotient_terms_calculated.append(quotient_term_result) # Store the actual (coeff, exp)

            # get_tex_mobjects_for_poly now returns a list, so take the first element
            new_quotient_mobj = self.get_tex_mobjects_for_poly([quotient_term_result])[0]

            # Position the new quotient term
            quotient_exp = quotient_term_result[1]
            new_quotient_mobj.move_to(RIGHT * self.exponent_x_map[quotient_exp] + UP * (self.division_bracket.get_y() + self.division_bracket.height/2 + 0.1))
            
            self.play(Write(new_quotient_mobj))
            self.quotient_mobjects.add(new_quotient_mobj)
            self.wait(0.5)

            # --- Multiply Quotient Term by Divisor ---
            product_coeffs = self.poly_multiply_term_by_poly(quotient_term_result, divisor_coeffs)
            
            # Use VGroup(*...) with arrangement for the product terms
            product_mobjects_list = self.get_tex_mobjects_for_poly(product_coeffs)
            product_mobjects = VGroup()
            
            # Position each product term correctly
            for mobj_prod_item in product_mobjects_list:
                p_exp = self.parse_tex_exponent(mobj_prod_item.get_tex_string())
                if p_exp in self.exponent_x_map:
                    mobj_prod_item.move_to(RIGHT * self.exponent_x_map[p_exp] + UP * self.current_y_pos_for_new_row)
                else:
                    # Fallback for unexpected exponent
                    if product_mobjects:
                        mobj_prod_item.next_to(product_mobjects[-1], RIGHT, buff=0.3)
                    else: # If it's the very first product term and its exp isn't mapped
                        mobj_prod_item.move_to(self.current_remainder_mobjects.get_x() + UP * self.current_y_pos_for_new_row)
                    mobj_prod_item.set_y(self.current_y_pos_for_new_row)
                product_mobjects.add(mobj_prod_item)
            
            self.play(TransformFromCopy(new_quotient_mobj, product_mobjects), run_time=1.5)
            self.wait(0.5)

            # --- Draw Subtraction Line ---
            subtraction_line = Line(LEFT, RIGHT)
            if product_mobjects:
                # Ensure line spans exactly where the product terms are
                leftmost_x = product_mobjects.get_x() - product_mobjects.get_width()/2
                rightmost_x = product_mobjects.get_x() + product_mobjects.get_width()/2
                subtraction_line.set_x((leftmost_x + rightmost_x) / 2)
                subtraction_line.set_width(rightmost_x - leftmost_x + 0.5) # Add some padding
            else:
                 subtraction_line.set_width(3).set_x(0) # Default if product is empty
            
            subtraction_line.next_to(product_mobjects, DOWN, buff=0.1)
            self.play(Create(subtraction_line))
            self.wait(0.5)

            # --- Animate Negation of Product Terms (for visual subtraction) ---
            negated_product_mobjects_for_display = VGroup()
            # Iterate through the original product_coeffs to create negated MathTex
            for (p_coeff, p_exp), mobj_orig in zip(product_coeffs, product_mobjects):
                new_str = self.poly_to_tex_str([(-p_coeff, p_exp)]) # Get LaTeX string for negated term
                negated_mobj = MathTex(new_str).move_to(mobj_orig.get_center()).set_color(BLUE)
                negated_product_mobjects_for_display.add(negated_mobj)
            
            self.play(FadeOut(product_mobjects), FadeIn(negated_product_mobjects_for_display), run_time=0.8)
            self.wait(0.5) # Hold the negated terms briefly
            
            # --- Calculate New Mathematical Remainder ---
            current_active_coeffs = self.normalize_poly(current_active_coeffs) # Ensure it's clean for subtraction
            subtraction_result_coeffs = self.poly_subtract(current_active_coeffs, product_coeffs)

            # --- Animate Cancellation and Dropping Down Terms ---
            animations = []
            
            # Exponents that are in the mathematical subtraction result
            exponents_in_subtraction_result = {exp for coeff, exp in subtraction_result_coeffs}

            # Fade out terms that cancel from the *current* remainder mobjects and the *negated* product mobjects
            for mobj in self.current_remainder_mobjects:
                try:
                    exp_from_mobj_str = self.parse_tex_exponent(mobj.get_tex_string())
                    # Check if this term's exponent is NOT in the final result (meaning it canceled)
                    if exp_from_mobj_str not in exponents_in_subtraction_result: 
                        animations.append(FadeOut(mobj, scale_factor=0.5))
                except ValueError:
                    pass

            for mobj in negated_product_mobjects_for_display:
                try:
                    exp_from_mobj_str = self.parse_tex_exponent(mobj.get_tex_string())
                    # If the negation caused a term to become zero after combining
                    if exp_from_mobj_str not in exponents_in_subtraction_result:
                         animations.append(FadeOut(mobj, scale_factor=0.5))
                except ValueError:
                    pass

            # Execute cancellation animations
            self.play(*animations, run_time=1)
            # Remove old mobjects from the scene
            self.remove(self.current_remainder_mobjects, negated_product_mobjects_for_display)

            # 2. Determine and animate "dropping down" terms from the original dividend.
            new_current_active_coeffs_for_next_step = list(subtraction_result_coeffs) # Start with current subtraction result
            
            dropped_terms_anims = []

            # Update which exponents from the original dividend have been processed
            for _, exp in current_active_coeffs:
                self.used_original_dividend_exponents.add(exp)
            
            # Iterate through original dividend terms from highest to lowest.
            # We want to drop terms that are 'next' in line after the current active calculation.
            for orig_d_coeff, orig_d_exp in self._original_dividend_full_terms:
                # Conditions for dropping down:
                # 1. The term's exponent is less than the leading exponent of the current calculation
                #    (meaning it's "below" the terms being actively processed).
                # 2. The term's exponent has not been "used" yet (i.e., not part of a previous active calculation or already dropped).
                # 3. The term's exponent is not already present in the `new_current_active_coeffs_for_next_step`
                #    (to avoid duplicates if it was a direct result of the subtraction).
                
                is_in_subtraction_result = False
                for c_n, e_n in subtraction_result_coeffs:
                    if e_n == orig_d_exp:
                        is_in_subtraction_result = True
                        break

                if (orig_d_exp < leading_active_term[1] and
                    orig_d_exp not in self.used_original_dividend_exponents and
                    not is_in_subtraction_result):
                    
                    # Add this term to the new active coefficients
                    new_current_active_coeffs_for_next_step.append((orig_d_coeff, orig_d_exp))
                    self.used_original_dividend_exponents.add(orig_d_exp) # Mark as used

            new_current_active_coeffs_for_next_step = self.normalize_poly(new_current_active_coeffs_for_next_step) # Re-normalize after adding dropped terms

            # 3. Create and animate the new remainder Mobjects
            # This is the visual representation of `new_current_active_coeffs_for_next_step`
            next_remainder_display_mobjects = VGroup()
            anims_for_new_remainder_creation = []

            # Get the MathTex objects for the new remainder from the updated coefficients
            new_remainder_terms_list = self.get_tex_mobjects_for_poly(new_current_active_coeffs_for_next_step)

            # Store a map of exponents to mobjects for easier positioning
            new_remainder_mobj_map = {self.parse_tex_exponent(mobj.get_tex_string()): mobj for mobj in new_remainder_terms_list}

            # Iterate through exponents from highest to lowest to ensure correct order in VGroup
            max_rem_exp = self.poly_degree(new_current_active_coeffs_for_next_step) if new_current_active_coeffs_for_next_step else -1
            
            # Determine min exponent needed for range (to include all relevant columns)
            min_exp_overall = -1 # Start from -1 to cover all exponents down to constant term (exp 0)
            if self._original_dividend_full_terms:
                min_exp_overall = min(exp for _,exp in self._original_dividend_full_terms)
            
            # Iterate through all relevant exponent positions to place terms
            for e in range(max(max_rem_exp, min_exp_overall), -1, -1):
                if e in new_remainder_mobj_map:
                    mobj = new_remainder_mobj_map[e]
                    target_pos = RIGHT * self.exponent_x_map[e] + UP * (self.current_y_pos_for_new_row - 0.7)
                    mobj.move_to(target_pos)
                    anims_for_new_remainder_creation.append(Create(mobj))
                    next_remainder_display_mobjects.add(mobj)
            
            # If the final mathematical remainder is '0' and nothing to drop, explicitly show '0'
            if not new_current_active_coeffs_for_next_step and not next_remainder_display_mobjects:
                final_zero_remainder = MathTex("0").move_to(RIGHT * self.exponent_x_map[0] + UP * (self.current_y_pos_for_new_row - 0.7))
                anims_for_new_remainder_creation.append(Create(final_zero_remainder))
                next_remainder_display_mobjects.add(final_zero_remainder)

            self.play(*dropped_terms_anims, *anims_for_new_remainder_creation, run_time=1.5)
            
            # Update `current_active_coeffs` for the next loop iteration (mathematical state)
            current_active_coeffs = new_current_active_coeffs_for_next_step
            # Update `self.current_remainder_mobjects` for the next step's visual reference
            self.current_remainder_mobjects = next_remainder_display_mobjects
            
            self.wait(1)

            self.current_y_pos_for_new_row -= self.current_remainder_mobjects.height * 1.5 # Adjust for next row

        # Store the final remainder
        self.final_remainder_coeffs = self.normalize_poly(current_active_coeffs)
        
        # Ensure the final remainder is visible if the loop ended and it's not empty
        # (It should be visible as `self.current_remainder_mobjects` was updated in the loop)
        # This check prevents adding '0' again if it was already added in the loop
        if not self.final_remainder_coeffs and (not self.current_remainder_mobjects or 
                                               (len(self.current_remainder_mobjects) == 1 and 
                                                self.current_remainder_mobjects[0].get_tex_string() != "0")):
             final_zero_mobj = MathTex("0")
             # Position the final zero correctly if it's the actual end
             if self.current_remainder_mobjects:
                 final_zero_mobj.move_to(self.current_remainder_mobjects.get_center()) # Overwrite current remainder
             else:
                 final_zero_mobj.move_to(RIGHT * self.exponent_x_map[0] + UP * (self.current_y_pos_for_new_row + 0.7))
             self.play(Create(final_zero_mobj))
             self.wait(1)
             self.current_remainder_mobjects = VGroup(final_zero_mobj) # Update for final display if needed


    def display_final_result(self):
        """Displays the final quotient and remainder, and the overall equation."""
        final_quotient_str_computed = self.poly_to_tex_str(self.quotient_terms_calculated)
        final_remainder_str = self.poly_to_tex_str(self.final_remainder_coeffs)

        quotient_display = MathTex(f"\\text{{Quotient: }} {final_quotient_str_computed}")
        remainder_display = MathTex(f"\\text{{Remainder: }} {final_remainder_str}")

        # Position final results
        result_group = VGroup(quotient_display, remainder_display).arrange(DOWN, buff=0.5).to_edge(DOWN)
        self.play(Write(result_group), run_time=2)
        self.wait(3)

    def parse_tex_exponent(self, tex_str):
        """
        A helper to extract exponent from a MathTex string.
        This is a heuristic and less robust than working directly with (coeff, exp) tuples.
        """
        cleaned_str = tex_str.replace('{', '').replace('}', '').strip('+').strip('-')
        
        if '^' in cleaned_str:
            try:
                return int(cleaned_str.split('^')[-1])
            except ValueError:
                pass
        if self.var in cleaned_str:
            return 1 # 'x' term (x^1)
        try:
            # Attempt to parse as a number; if successful, it's a constant term (exponent 0)
            float(cleaned_str)
            return 0 # Constant term (x^0)
        except ValueError:
            return -1 # Not a simple term, or parsing failed
