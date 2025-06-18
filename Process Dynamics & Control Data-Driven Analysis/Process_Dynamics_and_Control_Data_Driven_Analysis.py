import numpy as np
import matplotlib.pyplot as plt
from eng1014 import comp_trap_vector, heun
from datetime import datetime # import datetime model for datetime objects
######################========================
#custom function to parse datetime strings 
# --- Custom Counter Function (Replaces collections.Counter) ---
def custom_counter_most_common(data_list, n=1):
    """
    Counts occurrences of items in a list and returns the n most common.
    Replaces collections.Counter().most_common() functionality.
    """
    counts = {}
    for item in data_list:
        counts[item] = counts.get(item, 0) + 1

    # Sort items by count in descending order
    sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_items[:n]
def to_datetime_manual(date_string):
    """
    Converts a single date string to a datetime object.
    Attempts to parse with multiple common date-time formats.
    """
    if not isinstance(date_string, str) or not date_string.strip():
        return None # Return None if input is not a string or is empty/whitespace

    formats_to_try = [
        # YYYY-MM-DD HH:MM:SS (standard SQL format)
        '%Y-%m-%d %H:%M:%S',
        # DD/MM/YYYY HH:MM (common European/Australian format)
        '%d/%m/%Y %H:%M',
        # DD/MM/YY HH:MM (two-digit year, common in older systems)
        '%d/%m/%y %H:%M',
        # YYYY/MM/DD HH:MM:SS
        '%Y/%m/%d %H:%M:%S',
        # YYYY-MM-DD (date only)
        '%Y-%m-%d',
        # DD/MM/YYYY (date only)
        '%d/%m/%Y',
        # DD-MM-YYYY HH:MM:SS
        '%d-%m-%Y %H:%M:%S',
        # DD-MM-YYYY HH:MM
        '%d-%m-%Y %H:%M',
        # YYYY-MM-DDTHH:MM:SS (ISO 8601 with T separator)
        '%Y-%m-%dT%H:%M:%S',
        # MM/DD/YYYY HH:MM:SS (US format)
        '%m/%d/%Y %H:%M:%S',
        # MM/DD/YYYY HH:MM
        '%m/%d/%Y %H:%M',
    ]

    for fmt in formats_to_try:
        try:
            return datetime.strptime(date_string.strip(), fmt)
        except ValueError:
            continue # Try next format

    # If no format matches, return None
    return None 
####################=====================
#custom implementation ofr curve fit(simplified, only for linear and constant)
def custom_linear_fit(x,y):
    """Performs linear regression to find m and c for y=mx+c."""
    n=len(x)
    if n < 2:
        return np.nan, np.nan
    sum_x=np.sum(x)
    sum_y=np.sum(y)
    sum_xy=np.sum(x*y)
    sum_x2=np.sum(x**2)
    denominator=(n*sum_x2-sum_x**2)
    if denominator==0:
        return np.nan, np.nan #ewturn NaN for paraneters
    m=(n*sum_xy-sum_x*sum_y)/denominator
    c=(sum_y-m*sum_x)/n
    return m,c
def custom_constant_fit(x,y):
    """Find the mean of y for the constant fit y=c. """
    if len(y) == 0:
        return np.nan
    return np.mean(y)
def custom_curve_fit_logistic(x_data, y_data, p0=None):
    """A simplified logistic curve returns initial guesses."""
    #Filter out NaN from input data to ensure valid calculations
    valid_indices =~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_indices]
    y_data = y_data[valid_indices]
    if len(x_data) < 4 or len(y_data) < 4:
        return np.array([np.nan, np.nan,np.nan, np.nan]), None
    if p0 is None:
        L_init =(np.max(y_data) - np.min(y_data)) if len(y_data) > 1 else 1.0
        if len(y_data) > 1 and len(np.diff(y_data)) > 0:
            steepest_idx = np.argmax(np.abs(np.diff(y_data)))
            x0_init= x_data[steepest_idx]
        else:
            x0_init=np.mean(x_data) if len(x_data) > 0 else 0.0
        k_init = 0.5 #a reasonable starting guess for steepness
        b_init= np.min(y_data) if len(y_data) > 0 else 0.0
        p0 = [L_init, x0_init, k_init, b_init]
        return np.array(p0), None
#function to read csv file manually
# Function to read CSV manually, designed to handle quoted fields and BOM
############################+=======================

def read_csv_manual(filepath):
    """
    Reads a CSV file into a dictionary of lists (column_name: [values]).
    Handles quoted fields containing commas and the UTF-8 BOM.
    """
    data = {}
    with open(filepath, 'r', encoding='utf-8-sig') as f: # Use utf-8-sig to handle BOM
        lines = f.readlines()
        if not lines:
            return data

        # Custom parser for a single CSV line with quoted fields
        def parse_csv_line(line):
            fields = []
            current_field = []
            in_quote = False
            # Strip newline and carriage return characters from the end of the line
            line = line.strip('\n\r')
            for char in line:
                if char == '"':
                    in_quote = not in_quote
                elif char == ',' and not in_quote:
                    fields.append("".join(current_field).strip())
                    current_field = []
                else:
                    current_field.append(char)
            fields.append("".join(current_field).strip()) # Add the last field
            return fields

        # Process header
        # Apply .strip() to each element immediately after parsing to remove potential whitespace
        header = [h.strip() for h in parse_csv_line(lines[0])]
        for h in header:
            data[h] = []

        # Process rows
        for line_num, line in enumerate(lines[1:], start=2):
            values =  parse_csv_line(line)
            if len(values) != len(header):
                print(f"Warning: skipping malformed row {line_num} due to column count mismatch: {line.strip()}")
                continue
            for i, val in enumerate(values):
                data[header[i]].append(val)
    return data
#####################==============================
##############################################
#%% Cell 1  - Q1
#==================================
#Q1: Data inspection and cleaning
#================================
#load csv data fro, specified file path using manual reader
file_path = r"C:\Users\hp\OneDrive\Desktop\Keerthana\pH_neutralisation_Sept.csv"
raw_data_dict = read_csv_manual(file_path)
print(f"Loaded{len(list(raw_data_dict.values())[0]) if raw_data_dict else 0} raw data rows from CSV. ")
column_mapping = {
    "Date-Time (AEST/AEDT)": "Date-Time",
    "Inlet pH, [pH]": "Inlet pH",
    "Outlet pH, [pH]": "Outlet pH",
    "Outlet Flowrate, [L/min]": "Outlet Flowrate"
}
#robust data cleaning and tyoe conversion 
cleaned_dates=[]
cleaned_inlet_pH=[]
cleaned_flowrate=[]
cleaned_outlet_pH=[]
rows_skipped =0
num_raw_rows =len(raw_data_dict.get(list(raw_data_dict.keys())[0], []))#total rows read by read_csv+manual function
for r_idx in range(num_raw_rows):
    current_row_data={}
    row_is_valid=True
    skip_reason=""
    #PARSE DATE AND TIME
    date_str_key = "Date-Time (AEST/AEDT)"
    date_str=raw_data_dict.get(date_str_key, [])[r_idx]
    dt_obj= to_datetime_manual(date_str)
    if dt_obj is None:
        row_is_valid=False
        skip_reason=f"Date-Time parsing failed for '{date_str}'. "
    current_row_data["Date-Time"] =dt_obj
    numeric_original_cols=[k for k in column_mapping.keys() if k!= date_str_key]
    for original_col_name in numeric_original_cols:
        new_col_name= column_mapping[original_col_name]
        try:
            val_str=raw_data_dict.get(original_col_name, [])[r_idx]
            numeric_val=float(val_str)
            current_row_data[new_col_name]=numeric_val
        except (ValueError, IndexError) as e:
            row_is_valid=False
            skip_reason=f"Numeric parsing failed for '{original_col_name}' value '{val_str}' : {e}"
            break
    
    if row_is_valid:
        cleaned_dates.append(current_row_data["Date-Time"])
        cleaned_inlet_pH.append(current_row_data["Inlet pH"])
        cleaned_outlet_pH.append(current_row_data["Outlet pH"])
        cleaned_flowrate.append(current_row_data["Outlet Flowrate"])
    else:
        rows_skipped +=1

print(f"Skipped {rows_skipped} invalid rows during cleaning. ")
print(f"Retained {len(cleaned_dates)} valid rows after cleaning.")
#converting cleaned listed to numpy arrays 
dates=np.array(cleaned_dates, dtype=object)
inlet_pH=np.array(cleaned_inlet_pH, dtype=float)
outlet_pH=np.array(cleaned_outlet_pH, dtype=float)
flowrate=np.array(cleaned_flowrate, dtype=float)
##########################+==============================

#===================Q1(a): Flowrate correction=====================
#step 1: find all flowrate that are negative
neg_flow_indices=flowrate<0
neg_flow=flowrate[neg_flow_indices]
#step 2: round these negative values to 2 decimal places to simplify data
neg_rounded=np.round(neg_flow,2)
#step 3: find most common negative value
offset=0.0
if len(neg_rounded)>0:
    offset= custom_counter_most_common(neg_rounded.tolist(), 1)[0][0]
else:
    pass# no negative flow to correct
#step 4: correct all flowrate values 
corrected_flowrate = flowrate - offset
#step 5;l if after correction any flowrate value is less than 0.001 set them exactly zero
corrected_flowrate[corrected_flowrate<0.001]=0
#step 6: count how many values are zero after correction
zero_flow_count=np.sum(corrected_flowrate==0)
#print flowrate offset and zero flow readings we have
print("Q1(a): ")
print(f"Flowrate sensor offset: {offset} L/min")
print(f"numbers of zero_flow reading after correction: {zero_flow_count}")
#step 7: plot a histogram showing the distribution of the rounded negative flowrates 
plt.figure(1)
if len(neg_rounded) > 0:
    # Ensure bins cover the range from min to max with 0.01 step
    plt.hist(neg_rounded, bins=np.arange(np.min(neg_rounded), np.max(neg_rounded) + 0.01, 0.01), edgecolor='black')
else:
    plt.text(0.5, 0.5, "No negative flowrate data to plot", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.title("Figure 1: Histogram of Rounded Negative Flowrates")
plt.xlabel("Rounded Negative Flowrate (L/min)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
#########################=================================
#====================Q1(b): 3x1 plot of pH & Flowrate=============
#create a figure with 3 rows and 1 columns of plot
plt.figure(2, figsize=(12,10))
#subplot 1: plot oth inlet snmd outlet pH vs time(dates)
plt.subplot(3, 1, 1)
plt.plot(dates, inlet_pH, 'b.', markersize=2, label="Inlet pH")
plt.plot(dates, outlet_pH, 'r.', markersize=2, label="outlet pH")
plt.ylabel("pH")
plt.title("Figure 2a: Inlet and Outlet pH")
plt.legend()
plt.grid(True)
# Subplot 2: Plot the difference between outlet and inlet pH vs time
pH_diff = outlet_pH - inlet_pH
plt.subplot(3, 1, 2)
plt.plot(dates, pH_diff, 'g.', markersize=2, label="pH difference")
plt.ylabel("pH Difference")
plt.title("Figure 2b: pH Difference (Outlet - Inlet)")
plt.grid(True)
# Subplot 3: Plot the corrected flowrate vs time
plt.subplot(3, 1, 3)
plt.plot(dates, corrected_flowrate, 'k.', markersize=2, label="Corrected flowrate")
plt.ylabel("Flowrate (L/min)")
plt.xlabel("Date")
plt.title("Figure 2c: Corrected Outlet Flowrate")
plt.grid(True)
# Adjust layout to prevent overlapping plots
plt.tight_layout()
plt.show()
#######################==============================
# ===== Q1(c): Operating and Adjusting times =====

# System is operating if flowrate is greater than zero
operating = corrected_flowrate > 0
# System is adjusting if absolute difference between outlet and inlet pH is more than 0.5
adjusting = np.abs(pH_diff) > 0.5

# Count total number of minutes the system is operating and adjusting (each row = 1 minute)
total_operating = np.sum(operating)
total_adjusting = np.sum(adjusting)

print("\nQ1(c):")
print(f"Total minutes system is operating: {total_operating}")
print(f"Total minutes system is adjusting: {total_adjusting}")
#########################======================

#%% cell 2- Q2
#==================================
#Q2: Integration and cost estimation
#=========================
#==================Q2(a): Calculate dose rate R(t)====================
#target pH value set to 7(neutral)
target_pH=7
#convert target pH to hydrogen ion concetration(H+)
target_H=10**(-target_pH)
#convert inlet pH values to hydrogen ion concentration(H+)\
inlet_H=10**(-inlet_pH)
#calculate differebce in (H+) between target abd inlet(absolute value)
delta_H=np.abs(target_H-inlet_H)
#create an array to hold molecular weights cooresponding to acid/base dosing 
MW=np.zeros_like(inlet_pH)
#Assign molecular wight of acid(HCL=36.5g/mol) if inlet pH>8(meaning system dosing acid)
MW[inlet_pH>8]=36.5
#assign molecular weight of base(NaOH=40.0g/mol) if inlet pH<6
MW[inlet_pH<6]=40.0
#for pH between 6 and 8, no dosing is needed so molecular weight stays zero
#calculate the dose rate R(t) in grams oer minute:
#dose rate=flowrate(L/min)* concentration difference(mol/L)*molecular weight(g/mol)
R_t=corrected_flowrate*delta_H*MW #g/min
#plot dose rate over time
plt.figure(3)
plt.plot(dates, R_t, 'm.', markersize=2, label="Chemical dose rate")
plt.title("Figure 3: Chemical dose rate over time")
plt.xlabel("Date")
plt.ylabel("Chemical dose rate R(t) [g/mint]")
plt.grid(True)
plt.show()

#===============Q2(b): Integrate R(t) to get total mass================
#create an array representing each time point in minutes(assumes 1 minute intervals)
time_minute=np.arange(len(R_t))
#use the trapezoidal integration function to find the total chemical mass added over the entire time period
total_mass=comp_trap_vector(time_minute, R_t)#grans
print("\nQ2(b):")
print(f"Total chemical mass added: {total_mass:.2f} g")
#===============Q2(c): Daily dosing cost with tiered pricing===============
#extract the date part 'DD/MM/YYYY' for grouping
daily_doses={}
for i, date_obj in enumerate(dates):
    if date_obj is None:
        continue
    #sum R_t for eCH DATE
    date_only= date_obj.date()
    if date_only not in daily_doses:
        daily_doses[date_only]=0.0
    daily_doses[date_only]+=R_t[i]
#convert daily doses from grams to miligrams
daily_dose_mg_values=np.array(list(daily_doses.values()))*1000
#initialize total cost variable
cost_total=0
#calculate total cost based on the tiered pricing rules

for dose in daily_dose_mg_values:
    if dose<=500:
        #if dose is 500mg or less, cost is $0.02 per mg
        cost_total+=dose*0.02
    else:
        cost_total+=500*0.02+(dose-500)*0.015
print("\nQ2(c): ")
print(f"Total cost for the month: ${cost_total:.2f}")
###############################====================================
#%%Cell 3-Q3
#=======================
#Q3: Ceruve fitting and root finding
#=======================
#given titration data
volume = np.array([0, 0.5, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
pH = np.array([1.15, 1.21, 1.33, 1.6, 1.83, 2.57, 3.28, 4.13, 5.13, 6.26, 7.36, 9.14, 10.35, 11.13, 11.47, 11.72, 12.09, 12.2, 12.23, 12.33, 12.39, 12.5, 12.66])

#=====Defination model functions==========
#linear fucntion for region 1
def linear(x,m,c):
    return m*x+c
#logistic function for region 2(to medel the steep equivalance point region 
def logistic(x,L,x0,k,b):
    x=np.asarray(x, dtype=float)
    exp_arg=-k * (x-x0)
    return L/(1+np.exp(np.clip(exp_arg, -700, 700))) + b
#constant function for region 3(plateau)
def constant(x,c):
     return c+0*np.asarray(x)
#================Split data into region based on volume=============
#region 1: buffering region(pH rises gradually)
vol_region1=volume[volume<=12]
ph_region1=pH[volume<=12]
#region 2: equivalance region(steep rise)
vol_region2=volume[(volume>12)&(volume<=27)]
ph_region2=pH[(volume>12)&(volume<=27)]
#region 3: plateau region(pH stable after equivalence)
vol_region3=volume[volume>27]
ph_region3= pH[volume>27]

#==============fit region 1 with linear model ============
m1, c1, = custom_linear_fit(vol_region1, ph_region1)
popt1= [m1, c1]# mimic popt structure from scipy

#=========fit region 2 with logistic model ============

#find the colume where pH changes most rapidly. this is a rough estimate for x0

popt2_params, _ = custom_curve_fit_logistic(vol_region2, ph_region2)
L_fit, x0_fit, k_fit, b_fit = popt2_params
popt2 = [L_fit, x0_fit, k_fit, b_fit]

# Fit region 3 with constant model
c3 = custom_constant_fit(vol_region3, ph_region3)
popt3 = [c3]

# Prepare smooth x values for plotting fitted curves
x1 = np.linspace(np.min(vol_region1), np.max(vol_region1), 100) if len(vol_region1) > 1 else np.array([vol_region1[0]]) if len(vol_region1) > 0 else np.array([0])
x2 = np.linspace(np.min(vol_region2), np.max(vol_region2), 200) if len(vol_region2) > 1 else np.array([vol_region2[0]]) if len(vol_region2) > 0 else np.array([0])
x3 = np.linspace(np.min(vol_region3), np.max(vol_region3), 100) if len(vol_region3) > 1 else np.array([vol_region3[0]]) if len(vol_region3) > 0 else np.array([0])

# Compute fitted y-values
y1 = linear(x1, *popt1)
y2 = logistic(x2, *popt2) # Now using the approximated parameters
y3 = constant(x3, *popt3)

# Plot raw data and fitted curves
plt.figure(4, figsize=(10, 6))
plt.plot(volume, pH, 'ko', label="Raw data")
plt.plot(x1, y1, 'b-', label="Region 1: Linear fit")
plt.plot(x2, y2, 'g-', label="Region 2: Logistic fit (Approx.)")
plt.plot(x3, y3, 'r-', label="Region 3: Constant fit")
plt.xlabel("Volume of Base added (mL)")
plt.ylabel("pH")
plt.title("Figure 4: Fitted curve for strong acid-base reaction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#==================Equivalance point calculation=================
equivalence_volume=popt2[1] if len(popt2) > 1 else np.nan
print(f"Estimated point volume(approximate, without scipy.optimize.curve.fit): {equivalence_volume:.2f} mL")

#Q3(c)==========
#==============derivative plot for equivalence pount
#it calculates numerical derivative using central didderence where possible
d_pH_dV=np.diff(pH) /np.diff(volume)
vol_midpoints = (volume[:-1] + volume[1:]) / 2 
plt.figure(5, figsize=(10, 6))
plt.plot(vol_midpoints, d_pH_dV, 'go-', label="Numerical derivative")

plt.xlabel("Volume of Base added(mL)")
plt.ylabel("dpH/dV")
plt.title("Figure 5: derivative of pH with respect to volume)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#####################################



#%% Q4: ODEs - Torricells Law TAnk Draining
k=0.15 #sqrt(m/s)
h0 = 1.0 # initial height in m 
h_min = 0.01 #minimum height before stopping simulation
#define analystical and error functions before they are called
def analytical_drain_time(h0_val, h_val, k_val):
    '''Calculate theoretical drain time based on analytical solution. '''
    return (2 / k_val) * (np.sqrt(h0_val) - np.sqrt(h_val))
def percentage_error(t_numeric, t_exact):
    '''Calculated percentage error. '''
    if t_exact == 0:
        return 0 if t_numeric == 0 else float('inf') #avoid division by zero
    return abs((t_numeric - t_exact)/t_exact) * 100
# definr the ODE function dh/dt
def dhdt(t, h):
    '''ODE representing tank draining based on torricell’law. '''
    #ensure height does not go negative inside sqrt if simulation goes slightly below zero 
    if h<= h_min:
         return 0.0
    return -k * np.sqrt(h)
#modify the simulation function yo uder a more accurate solver(e.g Heun's method)
def simulate_tank_drain(dt, solver_func):
    #tSpan will be from 0 to an estimated large time to ensure it drains.
    #we will adjust the time vector after simulation to actual drain time. 
    #an initial guess for max time to ensure coverage, then trim.
    #a safe upper bound for drain time is (2/k)*sqrt(h0) for h=1, which is approx 14.4s
    #lets set a tSpan that is long enough, for example, 20s
    max_sim_time=analytical_drain_time(h0, 0.0, k) *2   #twice the time tp drain to 0 or (or a very small value close to 0)
    tSpan =[0,max_sim_time] #sufficiently large time span
    #use the chosen solver from eng1014.py
    # #Note: heun/midpount function returns t,y where y is dependent variabke(h here)
    t_raw, h_raw = solver_func(dhdt, tSpan, h0, dt)
    # find the actual drain time(when height drops below h_min)
    # #find the first index where h_raw <=h+min
    drain_idx_candidates = np.where(h_raw<=h_min)[0]
    if len(drain_idx_candidates)>0:
        first_drain_idx= drain_idx_candidates[0]
        if first_drain_idx > 0:
            #interpolate to find exact time h_main is reached
            t1, h1 = t_raw[first_drain_idx -1 ], h_raw[first_drain_idx - 1]
            t2, h2 = t_raw[first_drain_idx], h_raw[first_drain_idx]
            if (h2 - h1) != 0:
                #linear interpolation for drain time
                actual_drain_time = t1 + (h_min - h1) * (t2 -t1) /(h2 - h1)
            else:
                #if h1 and h2 are te same(plateau at h_main), use t1
                atual_drain_time = t2
        else: 
            actual_drain_time=t_raw[first_drain_idx]
        #trim the array to actual drain time
        valid_t_indices = t_raw <= actual_drain_time
        t_final = t_raw[valid_t_indices]
        h_final = h_raw[valid_t_indices]
        if len(t_final) ==0 or t_final[-1] < actual_drain_time:
            t_final = np.append(t_final, actual_drain_time)
            h_final = np.append(h_final, h_min)
        else:
            #if it didnot drain within tSpan, use the full arrays
            t_final= t_raw
            h_final=h_raw
            actual_drain_time=t_raw[-1]#report the end of the simulation time 
        return t_final, h_final, actual_drain_time


#Q4(a)
#using heun's method instead of custom Euler
times_01, heights_01, drain_time_01 = simulate_tank_drain(0.1, heun)
print(f"Q4(a): Drain time with dt=0.1s (heun's method):{drain_time_01:.2f} seconds ")

#Q4(b)
times_05, heights_05, drain_time_05 = simulate_tank_drain(0.5, heun)
times_10, heights_10, drain_time_10 = simulate_tank_drain(1.0, heun)
plt.figure(6, figsize=(10,6))
plt.plot(times_01, heights_01, label="dt = 0.1 s")
plt.plot(times_05, heights_05, label="dt = 0.5 s")
plt.plot(times_10, heights_10, label="dt = 1.0 s")
plt.xlabel("Time(s)")
plt.ylabel("Height(m)")
plt.title("Figure 5: Tank Height vs Time (heun's method)")
plt.legend()
plt.grid(True)
plt.show()
print("Q4(b):")
print(f"Drain time with dt= 0.5s(heun's method: {drain_time_05:.2f} seconds")
print(f"Drain time with dt= 1.s(heun's method: {drain_time_10:.2f} seconds")
print("Larger step sizes now provde better accureacy due to Heun's method (second_order).")

#Q4(c)
t_theory = analytical_drain_time(h0, h_min, k)
err_01 = percentage_error(drain_time_01, t_theory)
err_05 = percentage_error(drain_time_05, t_theory)
err_10 = percentage_error(drain_time_10, t_theory)
print("Q4(c): ")
print(f"theoretical drain time: {t_theory:.2f} seconds")
print(f"dt=0.1s → {drain_time_01:.2f}s({err_01:.2f}% error)")
print(f"dt=0.5s → {drain_time_05:.2f}s({err_05:.2f}% error)")
print(f"dt=1.0s → {drain_time_10:.2f}s({err_10:.2f}% error)")

#Q4(d)

dt_iter = 0.1# staring dt 
errors_log = []
steps_log = []
max_iterations = 100 #safety limit for the loop
current_iteration = 0

while current_iteration < max_iterations:
    _, _, t_num = simulate_tank_drain(dt_iter, heun)#use Heun's method here 
    err= percentage_error(t_num, t_theory)
    errors_log.append(err)
    steps_log.append(dt_iter)
    if err < 0.1: #target error less that 0.2%
        break
    dt_iter /= 2#halve the step size
    current_iteration +=1

print("Q4(d):")
if current_iteration < max_iterations:
    print(f"Max step size to achieve <0.1 % error (Heun's method): {dt_iter:.5f}s")
    print(f"final numerical drain time:{t_num:.5f}s")
    print(f"Corresponding percentage error: {err:.5f}%")
else:
    print("Could not achieve <0.1% error within maximum iterations. ")
    print(f"Smalles step sized tested: {dt_iter:.5f}s with error: {err:.5f}%")


#Q4(e)
plt.figure(7, figsize=(10,6))
plt.loglog(steps_log, errors_log, marker='o')
plt.xlabel("Time step size(s)")
plt.ylabel("Percentage error(%)")
plt.title("Figure 7: Log-log Plot of Error vs Step Size(Heun's Method)")
plt.grid(True, which="both", ls="--")
plt.show()

print("\nQ4(e): ")
print("The log-log plot now should show error ~ O(dt^2), indicating second-order convergance for Heun's method. ")


"""ENG1014 ASSIGNMENT – S1 2025 
pH neutralisation 
Due: 11:55 pm, Friday, Week 11 (23rd May) 
Late submissions: As per the Monash Marking and Feedback Procedure, a 5% penalty (-0.5 marks) per 
day, or part thereof, will be applied. Tasks submitted more than seven days after the due date without 
special consideration will not be marked.
Guidelines
This assignment is to be completed INDIVIDUALLY. Students are advised to review Monash University's
policies on academic integrity, plagiarism and collusion. Plagiarism occurs when you fail to acknowledge 
that the ideas or work of others are being used. Collusion occurs when you work in a manner not 
authorised by the teaching staff. Do not share your code with others. You may (and should) discuss general 
ideas and algorithms with your peers but the approach to coding must be your own. You must have a full 
understanding of it and be able to explain it during the interview session. All assignments will be 
checked using plagiarism and collusion detection software. In the event of suspected misconduct, the 
case will be reported to the Chief Examiner and the student's unit total may be withheld until the case has 
been decided. 
Instructions
Download the assignment materials from Moodle. Use the given template file ENG1014_Assignment.py
and modify the code within the template file. Stick to the template. Answer all questions as required. 
Check your solutions by running ENG1014_Assignment.py and ensuring the outputs for all questions are 
appropriately generated. The variables must be available in the pane after running the script, i.e. all output 
variables must be unique so that your demonstrator can examine them. 
This assignment assesses your ability to apply concepts taught in ENG1014. Coding is in many ways a 
creative pursuit, but to ensure that you can justify and explain your code and how it works, we strongly 
advise you to use functions and methods that you have learned in this unit. Therefore, you should NOT
use any external libraries, modules, or functions beyond those included in the ENG1014 coding 
environment, as specified in the beginning of the semester.
Submitting your assignment
Compress all required files into a single ZIP file to submit your assignment via Moodle. Name your ZIP file
as Assignment_ID, replacing ID with your student ID (e.g. Assigment_12345678.zip). Your ZIP file must 
include the following attachments: 
1. Your ENG1014_Assignment.py file containing all assignment tasks.
2. ALL additional files required by your .py file (including the data files, the ENG1014 module, and any other 
supplementary files you have written).
Your assignment will be assessed physically in Week 12. You are required to book a slot for the physical
interview session. Details about the slot booking will be sent out separately via Moodle Announcement. 
You must attend and be present for the assessment to be marked. You will receive a score of 0 if 
you do not attend an interview session. Your ZIP file will be downloaded from Moodle and only these 
files will be marked. Your demonstrator will extract (unzip) your submitted ZIP file and mark you based on 
the output of ENG1014_Assignment.py and your verbal explanations.
It is your responsibility to ensure that everything needed to run your solution is included in your 
ZIP file.
Marking Approach
This assignment is worth 10% of the unit mark. Your assignment will be graded using the following criteria: 
1. ENG1014_Assignment.py produces results automatically (additional user interaction
only if asked explicitly).
2. Your code produces correct results (printed values, plots, etc…) and is well written.
3. Poor programming practice (see table below) may result in a loss of up to 20%
4. Your ability to answer the demonstrator's questions in the Week 12 interview is where we test your 
understanding of the assignment questions and the submitted code. You may receive marks for correct 
explanations, even if your code is incorrect (and vice versa).
Hints
1. You may use any scripts that you have written in the labs and workshops. 
2. The tasks have been split into sub-questions. It is important to understand how each sub-question 
contributes to the whole, but each sub-question is effectively a stand-alone task that is part of the 
problem. Each can be tackled individually. 
3. It is recommended that you break down each sub-question into smaller parts too and figure out what 
needs to be done step-by-step. Then you put things together to complete the whole task or question. 
4. Attempt to solve the question or plan your algorithm by hand before attempting to code the solution.
Assignment Help
1. You can discuss the assignment with the unit coordinator (Dr. Lim) and your lab demonstrator (Ms. Teoh), 
however be aware that we will only answer general questions related to theory (e.g. “How can I 
determine if my step size is small enough when I’m solving an ODE?”) and not specific questions (e.g. “I’m 
stuck on question 3, how can I approach this?”).
Academic Integrity Expectations
1. You may discuss general ideas and approaches with peers. 
2. However, you should NOT write your code with a peer, nor share your code or sections of code directly 
with other students as this may constitute collusion.
3. AI & Generative AI tools MUST NOT BE USED within this assessment / task for the following 
reasons: This whole assessment task requires students to demonstrate human knowledge and 
skill acquisition without the assistance of AI.
Poor Programming Practices (PPP)
The table below summarises the criteria to avoid PPP deduction. A maximum deduction of 20% of the total 
assignment mark is applicable for PPP. Marks are deducted based on whether the criteria for each 
category is met. Remember to read the instructions carefully.
Category Criteria
1
Submission
a. Appropriate naming for the .py and .zip file submission
b. Include all supplementary files in submission
2
Documentation
a. Name, ID, Date on the submission
b. Communicate the answers for each question via either a plot or print statement
c. Include comments that briefly outline your code
d. Include a docstring for all user-defined functions
3
Coding
a. Use appropriate variable names
b. No overwriting of variable names
c. No hardcoding
d. Use efficient coding
4
Graphs
a. Include appropriate graph titles
b. Label x and y axes, including units where possible
c. Use appropriate and legible coloured lines and markers
d. Include a legend when there is more than one array plotted, including units where possible
ASSIGNMENT BRIEF 
Background
At the SAMPLE labs at Monash University, a wide range of research and teaching activities generate 
laboratory wastewater containing acids, bases, and chemical residues. To ensure compliance with 
environmental regulations and protect municipal drainage infrastructure, we use a pH neutralisation 
system to treat this wastewater before discharge.
Our pH neutralisation system continuously monitors the pH of incoming wastewater and automatically 
adjusts it using acid or base dosing to maintain a neutral range suitable for safe disposal. This ensures 
that effluent leaving the lab meets regulatory standards and minimises environmental impact. The system 
provides real-time data on inlet and outlet pH, offering valuable insights into the neutralisation process 
performance and chemical consumption. 
This assignment will use data from the pH neutralisation system as a case study to apply key numerical 
methods in engineering. The analysis will involve:
● Root Finding: Determining steady-state conditions or identifying critical points in pH response.
● Numerical Integration: Estimating total acid/base consumption over time or cumulative changes in 
pH.
● Ordinary Differential Equations (ODEs): Modelling the dynamic behaviour of the neutralisation 
process, including reaction kinetics and control system response.
Through these methods, students will gain practical experience in applying computational techniques to 
a real-world engineering system.
The SAMPLE pH Neutralisation System
Our system consists of four main components, shown in Figure 1.
1. Instrumentation for monitoring, controlling, and recording, including pH and ORP electrodes
2. Effluent holding tank
3. Chemical reagent storage tanks and addition pumps
4. Agitator(s)
The system automatically collects three data points in one-minute increments:
1. Inlet pH
2. Outlet pH
3. Outlet Flowrate [L/min]
System data for the month of September 2024 has been provided in the file 
“pH_neutralisation_Sept.csv”. 
Experimental lab data for the neutralisation of a strong base with a strong acid has been provided in the 
file “strong_acid_strong_base.csv”.
Figure 1: process flow diagram (PFD) of the SAMPLE pH neutralisation system
Assignment Objectives
In this assignment, you will apply core numerical methods in engineering—data cleaning, numerical 
integration, root finding, and ODE solving—to a real-world system: a pH neutralisation process used in 
laboratory wastewater treatment.
Working with real process data, you will simulate how engineers monitor and control chemical dosing to 
ensure safe, compliant effluent discharge. Throughout the assignment, you will work with actual process 
data, simulating how engineers monitor and control chemical dosing to maintain environmental 
compliance. Each question provides a different perspective on the system:
● Understanding sensor behaviour and flow conditions (Q1)
● Estimating cumulative chemical use and cost via integration (Q2)
● Modelling titration response curves for controller design (Q3)
● Predicting tank drain times using ODEs and assessing solution accuracy (Q4)
The assignment offers a practical taste of how numerical tools help engineers interpret data, make 
predictions, and support system design in real engineering contexts.
Q1. Data inspection and preparation
The dataset “pH_neutralisation_(Sept).csv” contains data from the pH neutralisation system for the 
month of September.
Sometimes a sensor isn't set up (calibrated) correctly, and this causes it to give wrong readings. In our 
case, the flowrate sensor shows some negative values, which don’t make physical sense because flowrate 
can’t be negative. This happens because of something called sensor offset. That means all the readings 
are shifted by a fixed amount. So even when the real flowrate is zero, the sensor shows a small negative 
number. To figure out how much the sensor is off, we look at all the negative flowrate values and find the 
most common one (this is called the mode). The mode is the sensor offset. Once we know the offset, we 
can fix the data by minus the mode from all the flowrate readings to get the true values, essentially 
increasing all the sensor readings to correct values.
(a) Identify all negative flowrate values and round the values to two decimal places. Determine the 
flowrate offset by creating a histogram on Figure (1) that shows the frequency of all rounded negative 
flowrate values. Apply this offset correction to all original flowrate values. Set any corrected flowrate values 
that are still below a threshold of 0.001 L/min to exactly zero (they represent non-flowing moments). Print 
the flowrate offset and number of zero flowrate values after correction to the console.
(b) On Figure (2), create a 3x1 subplot. In the first subfigure, plot the inlet pH and outlet pH as markers. 
In the second, plot the pH difference = outlet pH - inlet pH as markers. On the third, plot the cleaned outlet 
flowrate data from Q1(a). All three subfigures should be plotted against the date on the x-axis. 
Hint: If you cannot complete a task easily with the contents of the modules we have covered explicitly in ENG1014, 
remember that you are allowed to use any modules included in the standard Python installation. For example, 
there are components in modules that we haven’t discussed that can help you format time series plots.
Hint: Ensure that your code runs in a reasonable amount of time. If it seems to be taking a long time to run on
your machine, please consider a different solution to that task, even if it means you don’t get exactly the required
outputs. If we can’t run the code in a reasonable amount of time, we can’t mark it!
(c) We are interested in periods when the system is operating, defined as when the outlet flowrate is 
greater than 0, or adjusting, defined as when the absolute difference between outlet pH and inlet pH is 
greater than 0.5. Using your cleaned data from Q1(a), determine the total number of minutes when the 
system is operating and the total number of minutes when the system is adjusting. Print the total 
operating and adjusting times to the console.
Q2. Integration
pH is a way to describe how acidic or basic a solution is. A low pH means it's acidic (lots of H+ ions), and a 
high pH means it's basic (few H+ ions). 
To estimate how much acid or base is used in the neutralisation system over time, we can combine the 
outflow rate from the neutralisation system (how much treated water is leaving), and a dosing formula 
that tells us how much acid or base is needed based on how far the pH is from neutral. The system adds 
a strong acid when the pH is too high, and a strong base when the pH is too low, to bring the outlet pH 
into the target range of between 6-8.
Therefore, the system:
● Adds acid if inlet pH is above 8
● Adds base if inlet pH is below 6
● Does not dose if inlet pH is already between 6–8
The chemical dose rate is based on the relationship between pH and hydrogen ion concentration [H+], 
where:
[𝐻+] = 10−𝑝𝐻 Eq. 1
Therefore, the difference between the target and inlet hydrogen ion concentration, in mol/L, at any time, 
t is:
𝛥[𝐻 +](𝑡) = |10−𝑝𝐻𝑡𝑎𝑟𝑔𝑒𝑡 − 10−𝑝𝐻𝑖𝑛𝑙𝑒𝑡| Eq. 2
The required chemical dosing mass flowrate R(t) in g/min can be estimated by:
𝑅(𝑡) = 𝑄(𝑡) ⋅ 𝛥[𝐻 +](𝑡) ∙ 𝑀𝑊 Eq. 3
Where: 
● Q(t) is the outlet flowrate in L/min; 
● ∆[H+](t) is the difference between the target and inlet H+ concentration, which determines the 
acid or base addition required (Eq. 2).
● MW is the molecular weight of the acid or base, i.e. 36.5 g/mol for HCl (acid) or 40.0 g/mol for 
NaOH (base), respectively.
(a) Use your cleaned data from Question 1. Write a Python script that calculates the required dosing 
mass flowrate R(t) in g/min at each time point, based on the inlet pH and a fixed target pH of 7. Make sure 
no dosing occurs at pH between 6-8. Use your script to create an array of dosing mass flowrates and plot 
the resulting values against date on Figure (3).
(b) Use a method you have learned in ENG1014 to numerically integrate R(t) to determine the total 
amount of chemical added (in grams) over the entire month. Print the final result to the console.
(c) The lab pays for neutralising chemicals based on how much is used per day:
● For daily chemical use ≤ 500 mg, the cost is $0.02 per mg.
● For use > 500 mg, the cost is $0.02 per mg for the first 500 mg, and $0.015 per mg thereafter.
Calculate the total cost for the month using the tiered pricing scheme above. Print the final cost to the
console.
Q3. Curve Fitting and Root Finding
Accurately modelling the pH response of a neutralisation process is essential for tuning the control system 
that regulates chemical dosing. A well-fitted model allows us to predict how much acid or base needs to 
be added to reach a target pH, helping the system respond efficiently to varying inlet conditions. 
The dataset “strong_acid_strong_base.csv” contains a pH curve from a neutralisation process where a 
strong acid was added to a strongly basic effluent in the neutralisation tank. This dataset represents the 
measured pH of the solution as a function of the volume of base added in a laboratory titration 
experiment. The experimental pH data is typical, and exhibits three regions:
1. A slow increase (buffering region, below 22.50 mL)
2. A sharp rise (equivalence point region, 22.50 - 27.50 mL)
3. A plateau (post-neutralisation, above 27.50 mL)
(a) Using the curve fitting models that you have learned in ENG1014, create a piecewise model for each 
region for the pH curve between pH 2.1 and pH 13.5. Determine the parameters for each piecewise model. 
Plot your piecewise models and the original data together on Figure (4). Use different colours to clearly 
indicate the region of your fitted piecewise models. Provide justification of your model fittings by printing
comments (1-3 sentences) to the console.
(b) Create a function that accepts one or multiple values for volume of base added (mL), and/or other 
necessary parameters. The function should return the estimated pH between 2.1 and 13.5, calculated
using your piecewise models from Question 3(a). Use if-elif-else structures with logical conditions to apply 
the appropriate piecewise model to each volume. Print the predicted pH by your piecewise models for 
four dosing volume values of 22, 24, 26, and 28 mL, to the console.
(c) Use a numerical root-finding method to determine the volume of base required to reach neutral pH 
(pH = 7.0) based on your piecewise model. Print the volume of base added at pH = 7.0 and the 
corresponding pH value from your model to the console. Plot and label this point on Figure (4) as a red
circle.
Q4. ODEs
At the end of each month, the effluent holding tank in the pH neutralisation system must be emptied 
safely to allow for regular cleaning and maintenance. This is done by opening a drain valve at the bottom 
of the tank. Modelling the tank draining behaviour, i.e. the drain time of the tank, is useful for maintenance 
staff to plan their activities accordingly.
The flow rate through the valve at the tank outlet is governed by Torricelli’s Law: 
𝑑ℎ
𝑑𝑡
= −𝑘√ℎ Eq. 4
Where:
● h(t) = height of the fluid in the tank at time t [m]
● k = valve constant, based on the valve size and fluid properties [√𝑚/𝑠]
● t = time [s]
(a) Implement Euler’s method to solve the differential equation above, assuming the tank starts full with 
a liquid height of 1.0 m. Use a constant 𝑘 = 0.15 √𝑚/𝑠, and simulate the system until the height of the 
liquid drops below 0.01 m. Use time steps of 0.1 s for this initial simulation. Print the total time required 
to reduce the height of liquid in the tank to 0.01 m or less to the console.
(b) To investigate how the choice of time step affects the simulation, repeat Q4(a) using two additional 
time steps: 0.5 s and 1.0 s. Plot all three curves on a single graph in Figure (5) using different colours for 
each. Print the final drain time for each step size and a brief (1–2 sentences) summary of how the shape 
and accuracy of the solution changes as the step size changes to the console.
(c) The analytical solution to the ODE predicts the exact theoretical time required to drain the tank from 
an initial height to a final height. The relationship is:
𝑡𝑑𝑟𝑎𝑖𝑛 =
2
𝑘
(√ℎ0 − √ℎ) Eq. 5
Use this expression to calculate the theoretical drain time from 1.0 m to 0.01 m. Compare this value to the 
numerical results you obtained in Q4(b), and compute the percentage error for each time step. Print the 
theoretical drain time, all numerical drain times, and their corresponding percentage errors to the console.
(d) Write a script that uses a while loop to iteratively reduce the time step size (dt) and re-solve the ODE 
from Q4(a). Start from dt = 0.1 and halve the step size in each iteration (e.g., 0.1, 0.05, 0.025, ...). At each 
step, compare the numerical drain time to the analytical result from part (c). Continue looping until the 
percentage error between the numerical and analytical drain times is less than 0.1%. Print the maximum
step size required to achieve this accuracy, along with the final numerical drain time and corresponding 
percentage error.
(e) Using your results from Q4(d), create a log-log plot of percentage error versus time step size on Figure 
(6). Based on the slope of the log-log plot, print a short comment (1-2 sentences) to the console explaining 
how the gradient of the log-log plot reflects the error behaviour of Euler’s method.
End of Assignment"""
