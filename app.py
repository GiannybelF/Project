import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def euler_method(k, x0, y0, x_final, h):
    """
    Implements Euler's method for solving dy/dx = ky
    
    Parameters:
    k: constant in the differential equation
    x0: initial x value
    y0: initial y value
    x_final: final x value to compute to
    h: step size
    
    Returns:
    DataFrame with columns: Step, x, y, dy/dx, Analytical_y, Error
    """
    # Calculate number of steps
    n_steps = int((x_final - x0) / h)
    
    # Initialize arrays to store results
    steps = []
    x_values = []
    y_values = []
    derivatives = []
    analytical_values = []
    errors = []
    
    # Initial values
    x = x0
    y = y0
    
    for i in range(n_steps + 1):
        # Calculate derivative at current point
        dy_dx = k * y
        
        # Calculate analytical solution at current x
        y_analytical = y0 * math.exp(k * (x - x0))
        
        # Calculate error
        error = abs(y - y_analytical)
        
        # Store current values
        steps.append(i)
        x_values.append(x)
        y_values.append(y)
        derivatives.append(dy_dx)
        analytical_values.append(y_analytical)
        errors.append(error)
        
        # Calculate next y using Euler's method (except for last iteration)
        if i < n_steps:
            y = y + h * dy_dx
            x = x + h
    
    # Create DataFrame
    df = pd.DataFrame({
        'Step': steps,
        'x': x_values,
        'y (Euler)': y_values,
        'dy/dx': derivatives,
        'y (Analytical)': analytical_values,
        'Error': errors
    })
    
    return df

def validate_inputs(k, x0, y0, x_final, h):
    """
    Validates input parameters and returns error messages if any
    """
    errors = []
    
    if h <= 0:
        errors.append("Step size (h) must be positive")
    
    if x_final <= x0:
        errors.append("Final x value must be greater than initial x value")
    
    if (x_final - x0) / h > 10000:
        errors.append("Too many steps (>10000). Please increase step size or decrease range")
    
    return errors

def main():
    st.title("Euler's Method for Differential Equations")
    st.markdown("### Solving dy/dx = ky")
    
    st.markdown("""
    This application implements Euler's method to solve differential equations of the form **dy/dx = ky**, 
    where the rate of change of a quantity is directly proportional to the amount present.
    
    **Euler's Method Formula:** y_{n+1} = y_n + h × f(x_n, y_n)
    
    **Analytical Solution:** y = y₀ × e^(k×(x-x₀))
    """)
    
    # Create input form
    with st.form("euler_inputs"):
        st.subheader("Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            k = st.number_input(
                "Constant k in dy/dx = ky:",
                value=1.0,
                step=0.1,
                format="%.3f",
                help="The proportionality constant in the differential equation"
            )
            
            x0 = st.number_input(
                "Initial x value (x₀):",
                value=0.0,
                step=0.1,
                format="%.3f",
                help="Starting x coordinate"
            )
        
        with col2:
            y0 = st.number_input(
                "Initial y value (y₀):",
                value=1.0,
                step=0.1,
                format="%.3f",
                help="Starting y coordinate"
            )
            
            x_final = st.number_input(
                "Final x value:",
                value=2.0,
                step=0.1,
                format="%.3f",
                help="End point for calculation"
            )
        
        h = st.number_input(
            "Step size (h):",
            value=0.1,
            min_value=0.001,
            max_value=1.0,
            step=0.01,
            format="%.3f",
            help="Size of each step in Euler's method"
        )
        
        submitted = st.form_submit_button("Calculate Solution")
    
    if submitted:
        # Validate inputs
        errors = validate_inputs(k, x0, y0, x_final, h)
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Calculate solution
            try:
                df = euler_method(k, x0, y0, x_final, h)
                
                # Display results
                st.subheader("Solution Results")
                
                # Show final result
                final_euler = df.iloc[-1]['y (Euler)']
                final_analytical = df.iloc[-1]['y (Analytical)']
                final_error = df.iloc[-1]['Error']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Euler Approximation", f"{final_euler:.6f}")
                with col2:
                    st.metric("Analytical Solution", f"{final_analytical:.6f}")
                with col3:
                    st.metric("Absolute Error", f"{final_error:.6f}")
                
                # Display step-by-step solution table
                st.subheader("Step-by-Step Solution")
                
                # Format the dataframe for better display
                df_display = df.copy()
                for col in ['x', 'y (Euler)', 'dy/dx', 'y (Analytical)', 'Error']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.6f}")
                
                st.dataframe(df_display, use_container_width=True)
                
                # Create plot
                st.subheader("Visualization")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot Euler's method points
                ax.plot(df['x'], df['y (Euler)'], 'bo-', label='Euler Method', linewidth=2, markersize=4)
                
                # Plot analytical solution as a smooth curve
                x_smooth = np.linspace(x0, x_final, 1000)
                y_smooth = y0 * np.exp(k * (x_smooth - x0))
                ax.plot(x_smooth, y_smooth, 'r-', label='Analytical Solution', linewidth=2)
                
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f'Solution Comparison: dy/dx = {k}y, y({x0}) = {y0}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Error analysis
                st.subheader("Error Analysis")
                
                max_error = df['Error'].max()
                avg_error = df['Error'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Maximum Error", f"{max_error:.6f}")
                with col2:
                    st.metric("Average Error", f"{avg_error:.6f}")
                
                # Plot error
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(df['x'], df['Error'], 'g-', linewidth=2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('Absolute Error')
                ax2.set_title('Error vs x')
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig2)
                
                # Download option for results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"euler_method_results_k{k}_h{h}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"An error occurred during calculation: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.subheader("About Euler's Method")
    st.markdown("""
    **Euler's Method** is a first-order numerical procedure for solving ordinary differential equations (ODEs) 
    with a given initial value. It is the most basic explicit method for numerical integration of ODEs.
    
    **For the equation dy/dx = ky:**
    - This represents exponential growth (k > 0) or decay (k < 0)
    - Common applications include population growth, radioactive decay, and compound interest
    - The analytical solution is y = y₀ × e^(k×(x-x₀))
    
    **Accuracy considerations:**
    - Smaller step sizes generally produce more accurate results
    - However, very small step sizes may lead to computational errors
    - The method has local truncation error of O(h²) and global error of O(h)
    """)

if __name__ == "__main__":
    main()
