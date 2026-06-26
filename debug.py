import pandas as pd

# biermann event 8 exact window
peak_unix = int(pd.to_datetime("2018-09-11 23:43:42", utc=True).timestamp())
window_start = peak_unix - (0.1 * 3600.0) # 6 mins before
window_end = peak_unix + (24.0 * 3600.0)  # 24 hours after

print(f"--- BIERMANN EVENT 8 ---")
print(f"window start: {window_start}")
print(f"window end:   {window_end}\n")

# load your raw pipeline data
flares = pd.read_csv('new_detected_flares.csv')

# isolate just the star in question
suspect_flares = flares[flares['star_name'].str.contains("200759", na=False)]

print(f"found {len(suspect_flares)} flares for this star in your pipeline.")

for _, f in suspect_flares.iterrows():
    flare_t = float(f['time'])
    gap_hrs = (flare_t - peak_unix) / 3600.0
    
    print(f"\nflare detected at: {flare_t}")
    print(f"gap from peak: {gap_hrs:.2f} hours")
    
    if window_start <= flare_t <= window_end:
        print(">>> RESULT: inside window. the alias script failed.")
    else:
        print(">>> RESULT: outside window. the script worked, this is a genuinely novel flare.")
