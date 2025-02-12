
# Interactive_Map
## CSV Preparation
Here we use all the month data to filter out the fits files and concatenate them and put them in endfinal_individualfin1.csv 
```python
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import cKDTree

df = pd.read_csv("/content/endfinal_individualfin1.csv")
#df=df[400000:]
# Generate the bounding box and interpolate points
lat_min=df[["V0_lat", "V1_lat", "V2_lat", "V3_lat"]].min().min()
lat_max=df[["V0_lat", "V1_lat", "V2_lat", "V3_lat"]].max().max()
lon_min=df[["V0_lon", "V1_lon", "V2_lon", "V3_lon"]].min().min()
lon_max=df[["V0_lon", "V1_lon", "V2_lon", "V3_lon"]].max().max()

# Interpolate grid points at 0.01 precision
latitudes=np.arange(lat_min,lat_max,0.1)
longitudes=np.arange(lon_min,lon_max,0.1)
grid_points=np.array(np.meshgrid(latitudes,longitudes)).T.reshape(-1,2)

# Create a DataFrame for the grid
grid_df=pd.DataFrame(grid_points,columns=["latitude","longitude"])
tree=cKDTree(df[["V0_lat","V0_lon"]])
distances,indices=tree.query(grid_df[["latitude","longitude"]],k=1)

# Find nearest neighbors for the grid points
distances, indices = tree.query(grid_points)

# Assign values from nearest neighbors to new grid
grid_df = pd.DataFrame(grid_points, columns=["latitude", "longitude"])
grid_df["Si_area"] = df.iloc[indices]["Si_area"].values
grid_df["O/Si_ratio"] = df.iloc[indices]["O_area"].values / df.iloc[indices]["Si_area"].values
grid_df["Na/Si_ratio"] = df.iloc[indices]["Na_area"].values/ df.iloc[indices]["Si_area"].values
grid_df["Mg/Si_ratio"] = df.iloc[indices]["Al_area"].values/ df.iloc[indices]["Si_area"].values
grid_df["Al/Si_ratio"] = df.iloc[indices]["Mg_area"].values/ df.iloc[indices]["Si_area"].values
grid_df["Ca/Si_ratio"] = df.iloc[indices]["Ca_area"].values/ df.iloc[indices]["Si_area"].values
grid_df["Ti/Si_ratio"] = df.iloc[indices]["Ti_area"].values/ df.iloc[indices]["Si_area"].values
grid_df["Mn/Si_ratio"] = df.iloc[indices]["Mn_area"].values/ df.iloc[indices]["Si_area"].values
grid_df["Fe/Si_ratio"] = df.iloc[indices]["Fe_area"].values/ df.iloc[indices]["Si_area"].values
grid_df["O/Si_uncertainity"] = df.iloc[indices]["O/Si_uncertainty"].values
grid_df["Na/Si_uncertainity"] = df.iloc[indices]["Na/Si_uncertainty"].values
grid_df["Mg/Si_uncertainity"] = df.iloc[indices]["Mg/Si_uncertainty"].values
grid_df["Al/Si_uncertainity"] = df.iloc[indices]["Al/Si_uncertainty"].values
grid_df["Ca/Si_uncertainity"] = df.iloc[indices]["Ca/Si_uncertainty"].values
grid_df["Ti/Si_uncertainity"] = df.iloc[indices]["Ti/Si_uncertainty"].values
grid_df["Mn/Si_uncertainity"] = df.iloc[indices]["Mn/Si_uncertainty"].values
grid_df["Fe/Si_uncertainity"] = df.iloc[indices]["Fe/Si_uncertainty"].values
```
Here in the starting codeblock we use the endfinal_individualfin1 csv file which contains all the individual fitsfiles capable of producing desirable xrf lines. For interactive map we use the elemental ratios of all elements with Si as is done here and its saved to another dataframe grid df. In each row of the csv we get grid of 4 sets of lat and lon, so in each grid we consider all the points present and take them with a gap of 0.1 from min latlon to max latlon
```python
# Define the ratio columns and uncertainty columns
ratio_columns=[col for col in new_df.columns if 'ratio' in col]
uncertainty_columns=[col for col in new_df.columns if 'uncertainity' in col]
new_df[uncertainty_columns]=new_df[uncertainty_columns].fillna(0.00001)

# Fill NaN values in all other columns (except uncertainty) with 0
other_columns=[col for col in new_df.columns if col not in uncertainty_columns+['latitude', 'longitude']]
new_df[other_columns]= new_df[other_columns].fillna(0)
new_df
```
After removing all the Si area rows which captured a gaussian area of 0 during fitting, we fill the remaining Nan values with 0. Then we use the data to join the ratios with uncertainities for better visualisation.
```python
import pandas as pd

df=new_df1

#Parse the ± values into separate columns
ratio_columns=[col for col in df.columns if '/Si_ratio' in col]

#Create separate columns for mean and uncertainty
for col in ratio_columns:
    df[[f"{col}_mean", f"{col}_uncertainty"]] = df[col].str.split("±", expand=True).applymap(str.strip)
    df[f"{col}_mean"] = pd.to_numeric(df[f"{col}_mean"], errors='coerce')
    df[f"{col}_uncertainty"] = pd.to_numeric(df[f"{col}_uncertainty"], errors='coerce')

#Drop original combined columns to keep things clean
df.drop(columns=ratio_columns, inplace=True)

#Group by latitude and longitude, average values
mean_columns = [col for col in df.columns if '_mean' in col]
uncertainty_columns = [col for col in df.columns if '_uncertainty' in col]

# Group and aggregate
grouped_df = (
    df.groupby(['latitude', 'longitude'], as_index=False)
    .agg({**{col: 'mean' for col in mean_columns},
          **{col: 'mean' for col in uncertainty_columns}})
)

# Combine mean and uncertainty back into the original format
for col in mean_columns:
    original_col = col.replace("_mean", "")
    grouped_df[original_col] = (
        grouped_df[col].round(5).astype(str) + " ± " +
        grouped_df[col.replace("_mean", "_uncertainty")].round(5).astype(str)
    )

# Drop the temporary mean and uncertainty columns
grouped_df.drop(columns=mean_columns + uncertainty_columns, inplace=True)
grouped_df
```
Here we check for the duplicate coordinates which might have been present in each grid in the initial input csv taken essentially overlapping points and then we average them out to get the final_map_interactive_updated.csv which would be used for direct plotting.
## File Inputs
```python
lunar_map_path = '/content/drive/My Drive/lunar_map_resized_big.png'
csv_path = '/content/final_map_interactive_updated.csv'
```
Here we have taken the "final_map_interactive_updated csv" file for the plotting part which we prepared
```python
# Set scaling factor for figure size
scaling_factor = 0.4  # Adjust this as needed
fig_width = img_width * scaling_factor
fig_height = fig_width / (img_width / img_height)  # Maintain aspect ratio
```
Here as mentioned in the comments we take the scaling factor and check for the aspect ratio in the final plotting
```python
import plotly.graph_objects as go

# Create the Plotly figure with WebGL rendering
fig = go.Figure()

# Add the lunar map as the background
fig.add_layout_image(
    dict(
        source=lunar_map,
        x=0,
        y=img_height,
        xref="x",
        yref="y",
        sizex=img_width,
        sizey=img_height,
        xanchor="left",
        yanchor="top",
        layer="below",
    )
)

# Add scatter points using WebGL with hovertemplate and customdata
fig.add_trace(
    go.Scattergl(
        x=df['x_pixel'],
        y=df['y_pixel'],
        mode="markers",
        marker=dict(size=1, color='blue', opacity=0.25),
        hovertemplate=(
            "Lat: %{customdata[0]:.2f}<br>"
            "Lon: %{customdata[1]:.2f}<br>"
            "O/Si_ratio: %{customdata[2]}<br>"
            "Na/Si_ratio: %{customdata[3]}<br>"
            "Mg/Si_ratio: %{customdata[4]}<br>"
            "Al/Si_ratio: %{customdata[5]}<br>"
            "Ca/Si_ratio: %{customdata[6]}<br>"
            "Ti/Si_ratio: %{customdata[7]}<br>"
            "Mn/Si_ratio: %{customdata[8]}<br>"
            "Fe/Si_ratio: %{customdata[9]}<extra></extra>"
        ),
        customdata=df[[
            "latitude", "longitude", "O/Si_ratio", "Na/Si_ratio", "Mg/Si_ratio",
            "Al/Si_ratio", "Ca/Si_ratio", "Ti/Si_ratio", "Mn/Si_ratio", "Fe/Si_ratio"
        ]].values  # Pass raw data for hover
    )
)

# Correct aspect ratio and layout
aspect_ratio = img_width / img_height  # Calculate aspect ratio of the image

# Set figure width and height to maintain correct scaling
fig_width = img_width * 0.4
fig_height = fig_width / aspect_ratio

# Update layout for scaling and alignment
fig.update_layout(
    width=fig_width,
    height=fig_height,
    xaxis=dict(range=[0, img_width], visible=False),  # Match x-axis to image width
    yaxis=dict(range=[0, img_height], visible=False),  # Match y-axis to image height
    title="Interactive Lunar Map with Ratios",
)

# Lock axes scaling to maintain alignment
fig.update_yaxes(
    scaleanchor="x",  # Lock y-axis scaling to x-axis
    scaleratio=1      # 1:1 ratio for x and y
)

# Save the figure as an HTML file
output_file_path = '/content/drive/My Drive/interactive_map_last.html'
fig.write_html(output_file_path)

print(f"Interactive map saved at: {output_file_path}")

```
Here we have used plotly library and prepared the hovertemplate, added the lunar background and after running the code block we saved the code file in our drive for easy access purpose.





