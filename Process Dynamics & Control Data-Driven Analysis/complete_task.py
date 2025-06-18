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
    #Filter out NaNa from input data to ensure valid calculations
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
    max_sim_time=analytical_drain_time(h0, 0.0, k) *2   #twice the to,e tp drain to 0 or (or a very small value close to 0)
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