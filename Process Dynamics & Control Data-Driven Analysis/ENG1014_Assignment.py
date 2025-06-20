# ENG1014 Assignment 2025 Template File

# ENG1014_Assigment.py
# Student Name: Lee Zi Yuan
# Student ID: 34475990
# Date: 22/5/2025

# When all cells are run as written, this file should print all values and produce all plots required for the assignment.
# The cells where data should be imported and / or a figure generated are indicated below.

#%% Cell 0 - Import libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
import eng1014
from datetime import datetime

#%% Cell 1 - Q1
# Import pH_neutralisation_(Sept).csv

# Read CSV file
time = []
inlet_pH = []
outlet_pH = []
flowrate = []

# Open file
with open("pH_neutralisation_Sept.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        # Flexible time parsing
        def parse_datetime(t):
            for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
                try:
                    return datetime.strptime(t, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unknown time format: {t}")
        
        time.append(parse_datetime(row[0]))
        inlet_pH.append(float(row[1]))
        outlet_pH.append(float(row[2]))
        flowrate.append(float(row[3]))

# Convert to numpy arrays
time = np.array(time)
inlet_pH = np.array(inlet_pH)
outlet_pH = np.array(outlet_pH)
flowrate = np.array(flowrate)

# Q1(a): Determine sensor offset from negative values
neg_flows = flowrate[flowrate < 0]
rounded_neg_flows = np.round(neg_flows, 2)

# Plot figure 1
plt.figure(1)
plt.hist(rounded_neg_flows, bins=np.arange(min(rounded_neg_flows), 0.005, 0.002), edgecolor='black')
plt.title("Figure 1: Histogram of Negative Flowrate Values")
plt.xlabel("Flowrate (L/min)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()

# Find mode manually
unique_vals, counts = np.unique(rounded_neg_flows, return_counts=True)
offset = unique_vals[np.argmax(counts)]

# Apply offset correction
corrected_flowrate = flowrate + abs(offset)
corrected_flowrate[corrected_flowrate < 0.001] = 0

# Print Q1(a) results
print("1(a):")
print(f"Sensor offset: {offset:.2f} L/min")
print(f"Number of zero flowrate values after correction: {np.sum(corrected_flowrate == 0)}")

# Q1(b): Plot inlet/outlet pH, pH diff, and corrected flowrate
pH_diff = outlet_pH - inlet_pH

# Plot figure 2 subplot 1
plt.figure(2, figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(time, inlet_pH, 'bo', markersize=2, label='Inlet pH')
plt.plot(time, outlet_pH, 'ro', markersize=2, label='Outlet pH')
plt.ylabel("pH")
plt.legend()
plt.grid(True)

# Plot figure 2 subplot 2
plt.subplot(3, 1, 2)
plt.plot(time, pH_diff, 'go', markersize=2)
plt.ylabel("Outlet - Inlet pH")
plt.grid(True)

# Plot figure 2 subplot 3
plt.subplot(3, 1, 3)
plt.plot(time, corrected_flowrate, 'ko', markersize=2)
plt.xlabel("Date")
plt.ylabel("Flowrate (L/min)")
plt.grid(True)

plt.tight_layout()
plt.suptitle("Figure 2: System Monitoring Plots", y=1.02)

# Q1(c): Determine system operating and adjusting times
operating_mask = corrected_flowrate > 0
adjusting_mask = np.abs(pH_diff) > 0.5

operating_time = np.sum(operating_mask)
adjusting_time = np.sum(adjusting_mask)

# Print Q1(c) results
print("\n")
print("1(c):")
print(f"Total operating time (minutes): {operating_time}")
print(f"Total adjusting time (minutes): {adjusting_time}")

#%% Cell 2 - Q2
# Use your cleaned data from Q1

# Convert datetime objects to numerical minutes since first measurement
time_min = np.array([(t - time[0]).total_seconds() / 60 for t in time])

# Q2(a): Compute dosing mass flowrate R(t)
pH_target = 7
H_target = 10**(-pH_target)

# Initialize R_t array
R_t = np.zeros(len(inlet_pH))

for i in range(len(inlet_pH)):
    pH_in = inlet_pH[i]
    if pH_in > 8:
        # Acid addition case
        H_in = 10**(-pH_in)
        delta_H = abs(H_target - H_in)
        MW = 36.5  # HCl molecular weight
        R_t[i] = corrected_flowrate[i] * delta_H * MW
    elif pH_in < 6:
        # Base addition case
        H_in = 10**(-pH_in)
        delta_H = abs(H_target - H_in)
        MW = 40.0  # NaOH molecular weight
        R_t[i] = corrected_flowrate[i] * delta_H * MW
    else:
        # No dosing case (pH between 6-8)
        R_t[i] = 0

# Plot R(t) vs time
plt.figure(3)
plt.plot(time, R_t, 'b.', markersize=2)
plt.xlabel('Date')
plt.ylabel('Dosing Rate R(t) [g/min]')
plt.title('Figure 3: Dosing Rate Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Q2(b): Integrate R(t) over time to find total chemical added (grams)
# Ensure time_min is strictly increasing for integration
if not np.all(np.diff(time_min) > 0):
    # Remove duplicate time points if they exist
    _, unique_indices = np.unique(time_min, return_index=True)
    time_min = time_min[unique_indices]
    R_t = R_t[unique_indices]

total_grams = eng1014.comp_trap_vector(time_min, R_t)
print("\n")
print("2(b):")
print(f"Total chemical used in September: {total_grams:.2f} grams")

# Q2(c): Calculate daily cost using tiered pricing
# Create array of dates
dates = np.array([t.date() for t in time])
unique_dates = np.unique(dates)

total_cost = 0.0
for day in unique_dates:
    # Get indices for this day
    day_mask = dates == day
    daily_grams = np.sum(R_t[day_mask])
    daily_mg = daily_grams * 1000  # convert to mg
    
    if daily_mg <= 500:
        cost = daily_mg * 0.02
    else:
        cost = 500 * 0.02 + (daily_mg - 500) * 0.015
    
    total_cost += cost
    
print("\n")
print("2(c):")    
print(f"Total cost for the month: ${total_cost:.2f}")
#%% Cell 3 - Q3
# Import strong_acid_strong_base.csv

# Load titration data
titration_data = np.genfromtxt('strong_acid_strong_base.csv', delimiter=',', skip_header=1)
volumes = titration_data[:, 0]
pH = titration_data[:, 1]

# Region 1: below 22.5 mL
mask1 = volumes < 22.5
a0_1, a1_1, r2_1 = eng1014.linreg(volumes[mask1], pH[mask1])

# Region 2: 22.5 - 27.5 mL
mask2 = (volumes >= 22.5) & (volumes <= 27.5)
a0_2, a1_2, r2_2 = eng1014.linreg(volumes[mask2], pH[mask2])

# Region 3: above 27.5 mL
mask3 = volumes > 27.5
a0_3, a1_3, r2_3 = eng1014.linreg(volumes[mask3], pH[mask3])

# Plot data and piecewise models
plt.figure(4)
plt.plot(volumes, pH, 'o', markersize=3, label='Measured pH')

# Region 1
plt.plot(volumes[mask1], a1_1*volumes[mask1]+a0_1, 'r-', label='Region 1 fit')
# Region 2
plt.plot(volumes[mask2], a1_2*volumes[mask2]+a0_2, 'g-', label='Region 2 fit')
# Region 3
plt.plot(volumes[mask3], a1_3*volumes[mask3]+a0_3, 'b-', label='Region 3 fit')

plt.xlabel('Volume of base added (mL)')
plt.ylabel('pH')
plt.title('Figure 4: Titration Curve with Piecewise Linear Fits')
plt.legend()
plt.grid(True)
plt.show()

# Print model details
print("\n")
print("3(a):")
print(f"Region 1: pH = {a1_1:.3f}*V + {a0_1:.3f} (r2 = {r2_1:.3f})")
print(f"Region 2: pH = {a1_2:.3f}*V + {a0_2:.3f} (r2 = {r2_2:.3f})")
print(f"Region 3: pH = {a1_3:.3f}*V + {a0_3:.3f} (r2 = {r2_3:.3f})")


#%% Cell 4 - Q4
import numpy as np
import matplotlib.pyplot as plt
import eng1014

# --- (a) Euler’s method simulation ---
h1 = 0.1
k = 0.15  # valve constant [√m/s]
h0 = 1.0  # initial height [m]
h_threshold = 0.01

# ODE: dh/dt = -k*sqrt(h)
dydt = lambda t, h: -k * np.sqrt(max(h, 0))

# Integrate until h ≤ 0.01 m
t1, h1_vals = eng1014.euler(dydt, [0, 500], h0, h1)  # initial large tspan
idx1 = np.where(h1_vals <= h_threshold)[0][0]
t_drain_01 = t1[idx1]

print("\n")
print("4(a):") 
print(f"Time to drain to 0.01 m with h=0.1s: {t_drain_01:.2f} s")

# --- (b) Euler’s method with larger timesteps ---
# 0.5 s timestep
h2 = 0.5
t2, h2_vals = eng1014.euler(dydt, [0, 500], h0, h2)
idx2 = np.where(h2_vals <= h_threshold)[0][0]
t_drain_05 = t2[idx2]

# 1.0 s timestep
h3 = 1.0
t3, h3_vals = eng1014.euler(dydt, [0, 500], h0, h3)
idx3 = np.where(h3_vals <= h_threshold)[0][0]
t_drain_10 = t3[idx3]

# Plot all three
plt.figure(5)
plt.plot(t1[:idx1+1], h1_vals[:idx1+1], 'b-', label='Δt=0.1s')
plt.plot(t2[:idx2+1], h2_vals[:idx2+1], 'r-', label='Δt=0.5s')
plt.plot(t3[:idx3+1], h3_vals[:idx3+1], 'g-', label='Δt=1.0s')
plt.xlabel('Time (s)')
plt.ylabel('Liquid Height (m)')
plt.title('Figure 5: Tank Draining Curves')
plt.legend()
plt.grid(True)
plt.show()

print("\n")
print("4(b):")
print("Drain times:")
print(f"Δt=0.1s: {t_drain_01:.2f} s")
print(f"Δt=0.5s: {t_drain_05:.2f} s")
print(f"Δt=1.0s: {t_drain_10:.2f} s")
print("Observation: Larger step sizes give less accurate, faster-simulated results.")

# --- (c) Analytical solution ---
t_drain_analytical = 2/k * (np.sqrt(h0) - np.sqrt(h_threshold))
print(f"4(c): Analytical drain time: {t_drain_analytical:.2f} s")

# Percentage errors
err_01 = abs((t_drain_01 - t_drain_analytical)/t_drain_analytical) * 100
err_05 = abs((t_drain_05 - t_drain_analytical)/t_drain_analytical) * 100
err_10 = abs((t_drain_10 - t_drain_analytical)/t_drain_analytical) * 100

print("\n")
print("4(c):")
print("Percentage errors:")
print(f"Δt=0.1s: {err_01:.2f}%")
print(f"Δt=0.5s: {err_05:.2f}%")
print(f"Δt=1.0s: {err_10:.2f}%")

# --- (d) Adaptive timestep loop until error < 0.1% ---
dt = 0.1
err = 100.0
while err > 0.1:
    t_temp, h_temp = eng1014.euler(dydt, [0, 500], h0, dt)
    idx = np.where(h_temp <= h_threshold)[0][0]
    t_num = t_temp[idx]
    err = abs((t_num - t_drain_analytical)/t_drain_analytical) * 100
    if err > 0.1:
        dt /= 2

print("\n")
print("4(d):")
print(f"Maximum step size for error < 0.1%: {dt:.5f} s")
print(f"Numerical drain time: {t_num:.3f} s, Percentage error: {err:.5f}%")

# --- (e) Log-log plot of error vs timestep size ---
dt_values = []
err_values = []
dt_test = 0.1
while dt_test >= 0.001:
    t_temp, h_temp = eng1014.euler(dydt, [0, 500], h0, dt_test)
    idx = np.where(h_temp <= h_threshold)[0][0]
    t_num = t_temp[idx]
    err = abs((t_num - t_drain_analytical)/t_drain_analytical) * 100
    dt_values.append(dt_test)
    err_values.append(err)
    dt_test /= 2

plt.figure(6)
plt.loglog(dt_values, err_values, 'o-', label='Error vs. timestep')
plt.xlabel('Timestep size (s)')
plt.ylabel('Percentage Error (%)')
plt.title('Figure 6: Error Behavior of Euler’s Method')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

print("\n")
print("4(e):")
print("The log-log plot shows a linear trend with slope ≈ 1, indicating that Euler’s method has first-order error behavior (error ∝ timestep size).")
