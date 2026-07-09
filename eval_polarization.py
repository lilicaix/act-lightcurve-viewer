import pandas as pd
import numpy as np

from act_lightcurve_viewer import processing 

def main():    
    # load the transient data exported from run_pipeline.py
    try:
        flares_df = pd.read_csv('all_detected_flares.csv')
        coadded_thumbnails = np.load('coadded_thumbnails.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure run_pipeline.py has finished and saved these files first.")
        return
    
    # run the polarization limits function
    flares_list = flares_df.values.tolist()
        
    processing.calculate_polarization_limits(coadded_thumbnails, flares_list)
    pol_summary_df = pd.read_csv('polarization_summary.csv')    
    
    # load the cleaned biermann catalog
    biermann_df = pd.read_csv('biermann_events_clean.csv')

    biermann_cols_to_keep = ['star_name', 'event_id']
    biermann_subset = biermann_df[biermann_cols_to_keep].dropna(subset=['star_name']).copy()

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    def parse_name_to_coord(name):
        # convert both decimal-minute and arcsecond naming conventions into standard skycoords
        name = str(name).strip().replace('J', '')
        sign = '+' if '+' in name else '-'
        ra_part, dec_part = name.split(sign)
        
        ra_str = f"{ra_part[0:2]}h{ra_part[2:4]}m{ra_part[4:]}s"
        
        # check if the declination part uses decimal minutes (length 4 before the decimal)
        if '.' in dec_part and len(dec_part.split('.')[0]) == 4:
            dec_str = f"{sign}{dec_part[0:2]}d{dec_part[2:]}m"
        else:
            dec_str = f"{sign}{dec_part[0:2]}d{dec_part[2:4]}m{dec_part[4:]}s"
            
        return SkyCoord(f"{ra_str} {dec_str}", frame='icrs')

    # convert all string names into astropy skycoord objects
    pol_coords = SkyCoord([parse_name_to_coord(n) for n in pol_summary_df['Star Name']])
    biermann_coords = SkyCoord([parse_name_to_coord(n) for n in biermann_subset['star_name']])

    # find the nearest biermann coordinate for every pipeline star
    idx, d2d, _ = pol_coords.match_to_catalog_sky(biermann_coords)

    # matching radius = 60 arcsec
    max_sep = 60 * u.arcsec
    match_mask = d2d < max_sep

    # create the event_id column
    pol_summary_df['event_id'] = pd.Series(dtype='object')

    # apply the matching event ids only to the rows that passed the spatial threshold
    pol_summary_df.loc[match_mask, 'event_id'] = biermann_subset.iloc[idx[match_mask]]['event_id'].values

    # column reordering
    cols = pol_summary_df.columns.tolist()
    # move event id to 2nd column
    cols.insert(1, cols.pop(cols.index('event_id')))
    final_pol_df = pol_summary_df[cols]

    # remove letters in event id + rename column
    final_pol_df['event_id'] = final_pol_df['event_id'].str.replace(r'[a-zA-Z]', '', regex=True)
    final_pol_df.rename(columns={'event_id': 'Event ID'}, inplace=True)

    # save final summary table
    output_filename = 'polarization_summary.csv'
    final_pol_df.to_csv(output_filename, index=False)
    
if __name__ == "__main__":
    main()