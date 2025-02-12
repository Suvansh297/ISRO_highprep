
# Heatmap Generation

## Libraries Required
```python
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import random
```
These are the libraries used for the heatmap formation. Install Rasterio if not installed in local machine
```python
import pandas as pd

# Load the data
df = pd.read_csv("/content/endfinal_individualfin1.csv")

# Step 1: Add a column for total area of all elements
area_columns = [col for col in df.columns if "area" in col and col != "O_area"]
df["Total_area"] = df[area_columns].sum(axis=1)

# Step 2: Drop all uncertainty columns
uncertainty_columns = [col for col in df.columns if "uncertainity" in col]
df.drop(columns=uncertainty_columns, inplace=True)

# Step 3: Drop the oxygen-related column
df.drop(columns=["O_area"], inplace=True, errors="ignore")

# Step 4: Fill NaN values with 0
df.fillna(0, inplace=True)

# Step 5: Add columns for ratio of each element's area to total area
for col in area_columns:
    if col != "O_area":  # Ensure "O_area" is excluded
        df[f"{col}_ratio"] = df[col] / df["Total_area"]

# Step 6: Create a new DataFrame with only total area and element-area-ratio columns along with latitudes and longitudes
lat_lon_columns = [col for col in df.columns if "lat" in col or "lon" in col]
ratio_columns = [col for col in df.columns if "_ratio" in col]
new_df = df[lat_lon_columns + ["Total_area"] + ratio_columns]

# Save or display the resulting DataFrame
#new_df.to_csv("processed_data.csv", index=False)
new_df.head()
```
Here we use two csv files one contains added fits files and other one using individual fitsfiles, here latter is used. We dropped the oxygen column first and calculated weight percentage by taking sum of all elements gaussian area and obtaining individual element area with total area ratio and saving it in a new_df dataframe.

```python
# Load the base map
lunar_map_path = '/content/drive/My Drive/lunar_map_resized_big.png'
Image.MAX_IMAGE_PIXELS = None
lunar_map_pil = Image.open(lunar_map_path)
lunar_map = np.array(lunar_map_pil)
lunar_map = cv2.cvtColor(lunar_map, cv2.COLOR_RGB2BGR)
```
Here we use a resized lunar map for the heatmaps

```python
# Normalize the 'elemental_ratio' values
from matplotlib.cm import ScalarMappable
norm = Normalize(vmin=new_df['Ca_area_ratio'].min(), vmax=new_df['Ca_area_ratio'].max())

colormap = get_cmap('viridis')
scalar_map = ScalarMappable(norm=norm, cmap=colormap)
```
Here we individually normalize the ratios and use viridis colormap from matplotlib for the heatmap
```python
max_opacity=1.0
min_opacity=0.0
def lat_lon_to_pixel(lat, lon, img_width, img_height):
    x = int(((lon - min_lon) / (max_lon - min_lon)) * img_width)
    y = int(((max_lat - lat) / (max_lat - min_lat)) * img_height)
    return x, y

# Overlay the heatmap boxes
overlay = lunar_map.copy()
for _, row in new_df.iterrows():
    ul_x, ul_y = lat_lon_to_pixel(float(row['V0_lat']), float(row['V0_lon']), width, height)
    ur_x, ur_y = lat_lon_to_pixel(float(row['V1_lat']), float(row['V1_lon']), width, height)
    ll_x, ll_y = lat_lon_to_pixel(float(row['V2_lat']), float(row['V2_lon']), width, height)
    lr_x, lr_y = lat_lon_to_pixel(float(row['V3_lat']), float(row['V3_lon']), width, height)

    # Get box color
    normalized_value = norm(row['Ca_area_ratio'])
    color = scalar_map.to_rgba(row['Ca_area_ratio'], bytes=True)[:3]
    box_color = tuple(int(c) for c in color)
    d_opacity = min_opacity + normalized_value*(max_opacity - min_opacity)

    cv2.line(overlay, (ul_x, ul_y), (ur_x, ur_y), box_color, 2)
    cv2.line(overlay, (ur_x, ur_y), (lr_x, lr_y), box_color, 2)
    cv2.line(overlay, (lr_x, lr_y), (ll_x, ll_y), box_color, 2)
    cv2.line(overlay, (ll_x, ll_y), (ul_x, ul_y), box_color, 2)

# Blend the overlay with the base map
blended_image = np.zeros_like(lunar_map)
cv2.addWeighted(overlay, opacity, lunar_map, 1 - opacity, 0, blended_image)
lunar_map = blended_image[:, :, ::-1]  # Convert BGR to RGB for visualization
# Create the PNG with the color bar
fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(lunar_map)
ax.axis('off')

# Add the color bar
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # Adjust position
cb = plt.colorbar(scalar_map, cax=cbar_ax)
cb.set_label('Elemental Ratio')

# Save the PNG
output_png_path = '/content/drive/My Drive/Ca_coverage4_heatmap.png'
plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
plt.close()
```

Finally we get a visualization (for here calcium) area ratios on the taken lunar map by adding transparent, color-coded boxes to show the data. We calculated the box positions from latitude and longitude, with colors based on the scale and opacity we adjusted as 1.0 to match the ratios. Earlier we did include dynamic opacity which is stored in the d_opacity variable and it can be changed in the cv2.addWeighted function instead of opacity. We included a color bar for easy visualising. The heatmap was blended with the lunar map so that both the data and background are visible. In the end we saved the png file in our drive.


