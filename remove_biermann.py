import pandas as pd

def get_core_coord(name_str):
    """ extracts hhmm and ddmm from j-name, ignoring seconds/decimals """
    if 'J' not in str(name_str):
        return str(name_str)
    
    # get everything after 'J'
    coords = str(name_str).split('J')[1].strip()
    
    # grab ra hhmm and sign + dec ddmm
    if len(coords) >= 10:
        return coords[0:4] + coords[6:10]
    return coords

def main():
    # load datasets
    flares = pd.read_csv('new_detected_flares.csv')
    ref = pd.read_csv('biermann_events_clean.csv')

    # convert utc peak to unix seconds
    ref['peak_unix'] = pd.to_datetime(ref['peak_utc'], utc=True).astype('int64') // 10**9

    # convert rise/fall to seconds
    ref['rise_sec'] = ref['rise_hr'] * 3600.0
    ref['fall_sec'] = ref['fall_hr'] * 3600.0

    # set absolute unix boundaries
    ref['window_start'] = ref['peak_unix'] - ref['rise_sec']
    ref['window_end'] = ref['peak_unix'] + ref['fall_sec']
    
    # add core_coord for easy matching
    ref['core_coord'] = ref['star_name'].apply(get_core_coord)

    novel_flares = []
    removed = 0

    print("scanning for biermann matches...")
    
    # check flares against biermann windows
    for _, flare in flares.iterrows():
        flare_time = float(flare['time'])
        flare_star = str(flare['star_name']).strip()
        core_star = get_core_coord(flare_star)
        
        is_match = False
        
        # find matching core coordinates
        matching_events = ref[ref['core_coord'] == core_star]
        
        for _, b_event in matching_events.iterrows():
            # flag if within rise/fall window
            if b_event['window_start'] <= flare_time <= b_event['window_end']:
                is_match = True
                removed += 1
                print(f"removed: {flare_star} at {flare_time} (matches event {b_event['event_id']})")
                break 
                
        # keep if no match
        if not is_match:
            novel_flares.append(flare)

    # save completely new flares
    if novel_flares:
        final_df = pd.DataFrame(novel_flares)
        final_df.to_csv('sorted_new_flares.csv', index=False)
        print(f"\nsaved {len(final_df)} novel flares to 'sorted_new_flares.csv'")
    else:
        print("\nall flares matched biermann events. no novel flares found.")
        
    print(f"total removed: {removed}")

if __name__ == "__main__":
    main()