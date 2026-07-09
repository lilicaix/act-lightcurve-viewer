import pandas as pd
import numpy as np
import re
from astropy.coordinates import SkyCoord
import astropy.units as u

from act_lightcurve_viewer import file_io

def get_best_biermann_time(row):
    """returns the mjd time of the band with the highest snr."""
    snr_max = -1
    best_time_mjd = np.nan
    
    for band in ['220', '150', '090']:
        peak = row.get(f'f{band}_Peak')
        err = row.get(f'e_f{band}_Peak')
        tpeak = row.get(f'f{band}_tpeak')
        
        if pd.notna(peak) and pd.notna(err) and err > 0 and pd.notna(tpeak):
            snr = peak / err
            if snr > snr_max:
                snr_max = snr
                best_time_mjd = tpeak
                
    return best_time_mjd

def parse_iau_name_to_coord(name):
    """extracts mathematically readable coordinates from standard j-name strings."""
    # updated regex to catch optional decimals in both ra and dec just in case
    match = re.search(r'J(\d{6}\.?\d*)([+-]\d+\.?\d*)', str(name))
    if match:
        ra_raw, dec_raw = match.groups()
        
        # handle ra (hhmmss or hhmmss.s)
        if "." in ra_raw:
            ra_fmt = f"{ra_raw[0:2]}h{ra_raw[2:4]}m{ra_raw[4:]}s"
        else:
            ra_fmt = f"{ra_raw[0:2]}h{ra_raw[2:4]}m{ra_raw[4:6]}s"
        
        # handle dec (+ddmm.m or +ddmmss)
        if "." in dec_raw:
            dec_fmt = f"{dec_raw[0:3]}d{dec_raw[3:]}m"
        else:
            dec_fmt = f"{dec_raw[0:3]}d{dec_raw[3:5]}m{dec_raw[5:]}s"
            
        try:
            return SkyCoord(f"{ra_fmt} {dec_fmt}")
        except ValueError:
            return None
    return None

def main():
    print("loading data...")
    biermann_df = file_io.load_biermann_catalog('data/biermann_transient_candidates.txt')
    my_flares = pd.read_csv('all_detected_flares.csv')
    
    # initialize tracking columns
    my_flares['matched_to_biermann'] = False
    biermann_df['is_matched'] = False

    my_flares['biermann_event'] = 'None'
    
    print("converting iau strings to sky coordinates...")
    my_flares['skycoord'] = my_flares['star_name'].apply(parse_iau_name_to_coord)
    
    print(f"\n{'star name':<18} | {'seq':<4} | {'status'}")
    print("-" * 75)
    
    for idx, b_row in biermann_df.iterrows():
        b_name = str(b_row['Name']).strip()
        b_seq = str(b_row['Seq']).strip()
        
        b_time_mjd = get_best_biermann_time(b_row)
        if pd.isna(b_time_mjd):
            print(f"{b_name:<18} | {b_seq:<4} | missing time data")
            continue
            
        b_unix_time = (b_time_mjd - 40587.0) * 86400.0
        
        try:
            # biermann already provides decimal degrees!
            b_coord = SkyCoord(ra=b_row['RAdeg']*u.deg, dec=b_row['DEdeg']*u.deg)
        except Exception:
            continue
            
        # spatial matching (geometry based, ignores string names entirely)
        spatial_match_mask = pd.Series([False] * len(my_flares))
        min_sep = 999999
        for i, user_coord in my_flares['skycoord'].items():
            if user_coord is not None:
                sep = b_coord.separation(user_coord).arcmin
                if sep < min_sep:
                    min_sep = sep
                if sep < 3.0: 
                    spatial_match_mask[i] = True

                    my_flares.at[i, 'biermann_event'] = b_seq     
                    
        potential_matches = my_flares[spatial_match_mask]
        
        if len(potential_matches) == 0:
            print(f"{b_name:<18} | {b_seq:<4} | no spatial match (closest: {min_sep:.1f} arcmin away)")
            continue
            
        # time matching (1 hour window)
        time_diffs = (potential_matches['time'] - b_unix_time).abs()
        time_match_mask = time_diffs <= 3600.0
        
        if time_match_mask.any():
            print(f"{b_name:<18} | {b_seq:<4} | matched!")
            match_indices = potential_matches[time_match_mask].index
            my_flares.loc[match_indices, 'matched_to_biermann'] = True
            biermann_df.at[idx, 'is_matched'] = True
        else:
            min_hours_off = time_diffs.min() / 3600.0
            print(f"{b_name:<18} | {b_seq:<4} | location match, but time off by {min_hours_off:.1f} hours")
            
# process leftovers and strip letters
    leftovers = my_flares[~my_flares['matched_to_biermann']].drop(columns=['matched_to_biermann', 'skycoord'])
    leftovers['biermann_event'] = leftovers['biermann_event'].astype(str).str.replace(r'[a-zA-Z]', '', regex=True)
    leftovers.insert(1, 'biermann_event', leftovers.pop('biermann_event'))
    leftovers.to_csv('new_detected_flares.csv', index=False)

    # process matched and strip letters
    match_status = biermann_df[['Name', 'Seq', 'is_matched']].rename(columns={'Name': 'Biermann_Name', 'Seq': 'Sequence', 'is_matched': 'Status'})
    match_status['Sequence'] = match_status['Sequence'].astype(str).str.replace(r'[a-zA-Z]', '', regex=True)
    match_status['Status'] = np.where(match_status['Status'], 'matched', 'not detected')
    match_status.to_csv('matched_biermann_flares.csv', index=False)
    
    print("\ncross-match complete!")
    print(f"saved match status to 'matched_biermann_flares.csv'.")
    print(f"saved {len(leftovers)} novel flares to 'new_detected_flares.csv'.")

if __name__ == "__main__":
    main()