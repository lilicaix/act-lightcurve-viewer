import pandas as pd

def get_core_coord(name_str):
    """ extracts hhmm and ddmm from j-name, ignoring seconds/decimals """
    if 'J' not in str(name_str):
        return str(name_str)
    coords = str(name_str).split('J')[1].strip()
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
        # convert row to dictionary so we can easily add columns
        flare_data = flare.to_dict()
        flare_time = float(flare_data['time'])
        flare_star = str(flare_data['star_name']).strip()
        core_star = get_core_coord(flare_star)
        
        # set default tag
        flare_data['biermann_event'] = 'none'
        is_match = False
        
        # find matching core coordinates
        matching_events = ref[ref['core_coord'] == core_star]
        
        # --- NEW LOGIC: tag known stars before time check ---
        if not matching_events.empty:
            # grab all matched event ids and join them with a comma
            event_ids = matching_events['event_id'].astype(str).tolist()
            flare_data['biermann_event'] = ', '.join(event_ids)
        # ----------------------------------------------------
            
        for _, b_event in matching_events.iterrows():
            # flag if within rise/fall window
            if b_event['window_start'] <= flare_time <= b_event['window_end']:
                is_match = True
                removed += 1
                print(f"removed: {flare_star} at {flare_time} (matches event {b_event['event_id']})")
                break 
                
        # keep if no match
        if not is_match:
            novel_flares.append(flare_data)

    # save completely new flares
    if novel_flares:
        final_df = pd.DataFrame(novel_flares)
        # put the new column near the front for readability (optional, but clean)
        cols = final_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('biermann_event')))
        final_df = final_df[cols]
        
        final_df.to_csv('sorted_new_flares.csv', index=False)
        print(f"\nsaved {len(final_df)} novel flares to 'sorted_new_flares.csv'")
    else:
        print("\nall flares matched biermann events. no novel flares found.")
        
    print(f"total removed: {removed}")

if __name__ == "__main__":
    main()