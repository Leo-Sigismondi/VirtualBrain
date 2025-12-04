import mne
import os

def inspect_channels():
    # Path to a sample GDF file
    # Assuming the path from previous steps: data/raw/gdf/A01T.gdf
    file_path = "data/raw/gdf/A01T.gdf"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        # Try to find any gdf file
        import glob
        gdfs = glob.glob("data/raw/gdf/*.gdf")
        if gdfs:
            file_path = gdfs[0]
        else:
            print("No GDF files found.")
            return

    print(f"Inspecting file: {file_path}")
    try:
        raw = mne.io.read_raw_gdf(file_path, preload=False, verbose='ERROR')
        print(f"Total channels: {len(raw.ch_names)}")
        print("\nChannel List:")
        for i, name in enumerate(raw.ch_names):
            # MNE might guess type, but let's see the name
            print(f"Channel {i}: {name}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_channels()
