# High-Resolution Lunar Mapping

**Inter IIT Tech Meet 13.0 | Problem Statement 4**

## Project Overview

This repository contains our work on high-resolution elemental mapping of the lunar surface using X-ray fluorescence (XRF) data from the Chandrayaan-2 Large Area Soft X-ray Spectrometer (CLASS) instrument. Our project tackles the challenge of processing orbital XRF data to create detailed compositional maps that reveal the elemental distribution across the Moon.

## What We Built

We developed a complete pipeline for analyzing CLASS XRF data and visualizing lunar surface composition:

### 1. XRF Data Catalog
Starting with raw FITS files from ISRO's Pradan database (2021-2024), we built a comprehensive catalog of detected X-ray fluorescence lines. Our processing pipeline includes:

- **Smart signal classification** that separates useful solar flare events from background noise
- **Moving average technique** for robust background estimation instead of static calculations
- **Custom Gaussian fitting** tailored for each element, solving tricky cases like oxygen detection at low-energy channels
- **Overlap handling** between adjacent spectral lines with proper uncertainty quantification

The challenge with oxygen was particularly interesting—its characteristic K-line sits right at channel 37-38 where instrumental noise is high. We developed an analytical mirroring approach to get reliable gaussian fits despite the interference.

### 2. Elemental Base Maps
We mapped the spatial distribution of key elements (Mg, Al, Si, Ca, Ti, Fe, Na) across the lunar surface. The results validate known geochemical patterns:

- Clear **anti-correlation between Aluminum and Magnesium/Iron**, reflecting the fundamental difference between feldspathic highlands and mafic mare regions
- Strong concentration of mafic elements in **Oceanus Procellarum** (Ocean of Storms), consistent with its volcanic history and thin crust
- Silicon distributed evenly across the surface, serving as our normalization reference

Our heat maps use normalized photon flux data, with color grading optimized to highlight compositional variations across different lunar terrains.

### 3. Interactive Lunar Map
The centerpiece is an interactive visualization that consolidates all XRF analysis results. Built with Plotly and WebGL rendering for smooth performance, the map features:

- All data points overlaid on a lunar albedo basemap
- Hover functionality showing elemental ratios, weight percentages, and coordinates for each 12.5×12.5 km observation area
- **Sub-pixel resolution** where overlapping observation boxes create averaged properties for finer spatial detail

Due to the dataset size, the interactive map is served as an HTML file requiring significant RAM for optimal viewing.

### 4. Compositional Group Classification
Using calculated elemental ratios, we identified mineral composition groups across the mapped regions:
- Pyroxene (Ca-Fe-Mg silicates)
- Olivine (Mg-Fe silicates)  
- Plagioclase feldspar (Ca-Na-Al silicates)
- Ilmenite (Fe-Ti oxides)

We score potential mineral matches by comparing observed ratios against reference compositions, with lower standard deviation indicating better fits.

## Technical Approach

**Data Processing:** Python-based workflow using Pandas, NumPy, and Rasterio for handling FITS files and spatial data. XML metadata parsing for coordinates and timestamps, cross-referenced with XSM solar flare records.

**Spectral Analysis:** Energy channels converted from raw counts using gaussian decomposition. Background-subtracted spectra fitted with customizable masking regions per element. Area under curves calculated with overlap corrections.

**Mapping:** Pixel coordinates dynamically calculated from lat/lon to align with lunar basemap dimensions. Multiple observation coverage creates enhanced resolution through averaging.

**Visualization:** Plotly Scattergl for GPU-accelerated rendering of large datasets. Custom hover templates for efficient data retrieval without layout overhead.

## Key Results

- **XRF line coverage mapping** showing CLASS instrument coverage approaching 95% of the lunar surface (99% when including all data, focused on high-quality flare events for elemental detection)
- **Weight percentage calculations** for major rock-forming elements normalized against silicon
- **Validation of known lunar geochemistry**, including the mafic-feldspathic divide and volcanic mare compositions
- **Sub-pixel resolution enhancement** through overlapping observation analysis

## Repository Contents

```
├──Elemental abundances/   # The code which were used for catalogue detection
├── Interactive Map/       # HTML interactive visualization files
├── Lunar Basemaps/        # Generated elemental distribution heat maps
├── Report and Journal/    # Full technical report and documentation
└── README.md              # This file
```

Note: Due to repository size constraints, the full dataset and some processing outputs are not included directly. Key visualizations and results are provided in the folders above.

## Future Directions

The report outlines several promising extensions:
- **Deep learning super-resolution** using SRGAN methods to push beyond instrument pixel limits
- **Multi-frame super-resolution** approaches similar to LRO mission data processing
- **Machine learning for signal classification** using supervised methods from particle physics

These techniques could significantly enhance the resolution of CLASS data, potentially revealing compositional variations at scales finer than the native 12.5 km observation footprint.

## About

This project was developed for the Inter IIT Tech Meet 13.0, representing IIT Bhubaneswar's response to ISRO Problem Statement 4. The work demonstrates practical application of XRF spectroscopy principles to planetary science, combining data processing, spectral analysis, and interactive visualization to make lunar composition data accessible and interpretable.

---

**IIT Bhubaneswar | November 2024**
