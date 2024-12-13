import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Pile Capacity Calculator", layout="wide")


# --- Initialize Session State ---
def initialize_session_state():
    if "units" not in st.session_state:
        st.session_state.units = [
            {
                "name": "Unit 1",
                "top_depth": 0.0,
                "skin_friction": 0.0,
                "end_bearing": 0.0,
            }
        ]
    # Initialize other parameters
    if "diameter_min" not in st.session_state:
        st.session_state.diameter_min = 0.3
    if "diameter_max" not in st.session_state:
        st.session_state.diameter_max = 1.2
    if "length_min" not in st.session_state:
        st.session_state.length_min = 10
    if "length_max" not in st.session_state:
        st.session_state.length_max = 40
    if "reduction_factor" not in st.session_state:
        st.session_state.reduction_factor = 1.0
    if "three_d_embed" not in st.session_state:
        st.session_state.three_d_embed = False
    if "skin_friction_only" not in st.session_state:
        st.session_state.skin_friction_only = False


initialize_session_state()


# --- Helper Functions ---
def rename_units():
    """Ensure units are always sequentially named."""
    for i, u in enumerate(st.session_state.units, start=1):
        u["name"] = f"Unit {i}"


def load_csv(file):
    """Load parameters and units from uploaded CSV file into session state."""
    try:
        # Read the entire file as text
        file.seek(0)
        content = file.read().decode("utf-8")
        lines = content.splitlines()

        # Separate parameters and units
        params = {}
        units = []
        mode = "parameters"  # Start with parameters
        skip_next_line = False  # To skip the units header

        for line in lines:
            if line.strip() == "":
                mode = "units"
                skip_next_line = True  # Next line is header, skip it
                continue
            if mode == "parameters":
                if "," in line:
                    key, value = line.split(",", 1)
                    params[key.strip()] = value.strip()
            elif mode == "units":
                if skip_next_line:
                    # Skip the units header
                    skip_next_line = False
                    continue
                parts = line.split(",")
                if len(parts) == 4:
                    name, top_depth, skin_friction, end_bearing = parts
                    try:
                        units.append(
                            {
                                "name": name.strip(),
                                "top_depth": float(top_depth.strip()),
                                "skin_friction": float(skin_friction.strip()),
                                "end_bearing": float(end_bearing.strip()),
                            }
                        )
                    except ValueError as ve:
                        st.warning(f"Skipping unit due to conversion error: {ve}")

        # Update parameters
        if "diameter_min" in params:
            st.session_state.diameter_min = float(params["diameter_min"])
        if "diameter_max" in params:
            st.session_state.diameter_max = float(params["diameter_max"])
        if "length_min" in params:
            st.session_state.length_min = int(float(params["length_min"]))
        if "length_max" in params:
            st.session_state.length_max = int(float(params["length_max"]))
        if "reduction_factor" in params:
            st.session_state.reduction_factor = float(params["reduction_factor"])
        if "three_d_embed" in params:
            st.session_state.three_d_embed = (
                params["three_d_embed"].strip().lower() == "true"
            )
        if "skin_friction_only" in params:
            st.session_state.skin_friction_only = (
                params["skin_friction_only"].strip().lower() == "true"
            )

        # Update units
        if units:
            st.session_state.units = units
            rename_units()
        else:
            st.warning("No geological units found in the uploaded CSV.")

    except Exception as e:
        st.error(f"Failed to load CSV: {e}")


def generate_csv():
    """Generate CSV of current parameters and units."""
    output = io.StringIO()
    # Write parameters
    output.write("parameter,value\n")
    output.write(f"diameter_min,{st.session_state.diameter_min}\n")
    output.write(f"diameter_max,{st.session_state.diameter_max}\n")
    output.write(f"length_min,{st.session_state.length_min}\n")
    output.write(f"length_max,{st.session_state.length_max}\n")
    output.write(f"reduction_factor,{st.session_state.reduction_factor}\n")
    output.write(f"three_d_embed,{st.session_state.three_d_embed}\n")
    output.write(f"skin_friction_only,{st.session_state.skin_friction_only}\n\n")

    # Write units
    output.write("name,top_depth,skin_friction,end_bearing\n")
    for u in st.session_state.units:
        output.write(
            f"{u['name']},{u['top_depth']},{u['skin_friction']},{u['end_bearing']}\n"
        )

    return output.getvalue()


def calculate_3d_embedment_eb(toe_depth, diameter, sorted_units):
    """
    Calculate the effective end bearing considering 3D embedment rules and transitions.
    Rules:
    1. Start from the top of the pile and assign the end bearing of the first unit the pile enters.
       Initially, no end bearing is mobilized until 3 × D embedment is achieved into that first unit.
       Once the pile toe is embedded at least 3D into the first unit, that unit’s EB is fully available.

    2. When the pile passes from one unit (Unit A) into a deeper unit (Unit B):
       - If EB_B > EB_A and EB_B > 0:
         - For the first 3D within Unit B: use EB_A
         - After 3D: use EB_B
       - Else (EB_B ≤ EB_A or EB_B = 0):
         - For the first 3D within Unit B: use 0
         - After 3D: use EB_B

    3. If three_d_embed is False, simply take the end bearing of the unit where the toe lies.
    """
    embed_length = 3 * diameter
    previous_EB = 0.0
    current_EB = 0.0

    for i, unit in enumerate(sorted_units):
        unit_top = unit["top_depth"]
        unit_EB = unit["end_bearing"]
        if i < len(sorted_units) - 1:
            unit_bottom = sorted_units[i + 1]["top_depth"]
        else:
            unit_bottom = float("inf")

        if toe_depth < unit_top:
            # Toe is above this unit
            break

        if toe_depth >= unit_top and toe_depth < unit_bottom:
            # Toe is within this unit
            embed_depth = toe_depth - unit_top
            if embed_depth < embed_length:
                if unit_EB > previous_EB and unit_EB > 0:
                    effective_EB = previous_EB
                else:
                    effective_EB = 0.0
            else:
                if unit_EB > previous_EB and unit_EB > 0:
                    effective_EB = unit_EB
                else:
                    effective_EB = unit_EB
            return effective_EB
        elif toe_depth >= unit_bottom:
            # Toe is below this unit, check if 3D embedment was achieved
            embed_in_unit = unit_bottom - unit_top
            if embed_in_unit >= embed_length:
                if unit_EB > previous_EB and unit_EB > 0:
                    previous_EB = unit_EB
                else:
                    previous_EB = unit_EB
            else:
                if unit_EB > previous_EB and unit_EB > 0:
                    if embed_in_unit >= embed_length:
                        previous_EB = unit_EB
                    else:
                        # Not enough embedment, keep previous_EB
                        pass
                else:
                    previous_EB = unit_EB
    # If toe_depth exceeds all unit bottoms
    return previous_EB


def calculate_capacity(
    diameter, length, units, reduction_factor, three_d_embed, skin_friction_only
):
    """
    Calculate the pile capacity (in kN).
    """
    area_base = np.pi * (diameter / 2) ** 2  # m²
    perimeter = np.pi * diameter  # m

    # Sort units by top_depth
    sorted_units = sorted(units, key=lambda u: u["top_depth"])
    if sorted_units and sorted_units[0]["top_depth"] > 0:
        sorted_units.insert(
            0,
            {
                "name": "Dummy",
                "top_depth": 0.0,
                "skin_friction": 0.0,
                "end_bearing": 0.0,
            },
        )

    for i in range(len(sorted_units)):
        if i < len(sorted_units) - 1:
            sorted_units[i]["bottom_depth"] = sorted_units[i + 1]["top_depth"]
        else:
            sorted_units[i]["bottom_depth"] = float("inf")

    # Skin friction
    increment = 0.5  # meters
    segments = int(np.ceil(length / increment))
    total_friction = 0.0
    for seg in range(segments):
        seg_top = seg * increment
        seg_bot = min((seg + 1) * increment, length)
        seg_mid = (seg_top + seg_bot) / 2
        seg_len = seg_bot - seg_top

        current_unit = None
        for unit in sorted_units:
            if unit["top_depth"] <= seg_mid < unit["bottom_depth"]:
                current_unit = unit
                break
        if current_unit:
            sf = current_unit["skin_friction"]
            seg_friction = sf * perimeter * seg_len
            total_friction += seg_friction

    # End bearing
    end_bearing = 0.0
    if not skin_friction_only:
        toe_depth = length
        toe_unit = None
        for unit in sorted_units:
            if unit["top_depth"] <= toe_depth < unit["bottom_depth"]:
                toe_unit = unit
                break
        if not toe_unit and sorted_units:
            toe_unit = sorted_units[-1]

        if toe_unit:
            if three_d_embed:
                # Use the advanced 3D embedment logic
                effective_EB = calculate_3d_embedment_eb(
                    toe_depth, diameter, sorted_units
                )
            else:
                # No 3D embedment: just use toe_unit EB
                effective_EB = toe_unit["end_bearing"]
            end_bearing = effective_EB * area_base

    total_capacity = (total_friction + end_bearing) * reduction_factor
    return total_capacity


# --- Data Management Section ---
st.sidebar.header("Data Management")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    load_csv(uploaded_file)
    rename_units()

# Download CSV
csv_data = generate_csv()
st.sidebar.download_button(
    label="Download CSV Template",
    data=csv_data,
    file_name="pile_parameters_template.csv",
    mime="text/csv",
)

# --- Input Parameters ---
st.sidebar.header("Input Parameters")

# Pile Diameter Range Slider
diameter_range = st.sidebar.slider(
    "Diameter Range (m)",
    min_value=0.3,
    max_value=2.1,
    value=(st.session_state.diameter_min, st.session_state.diameter_max),
    step=0.3,
)
st.session_state.diameter_min, st.session_state.diameter_max = diameter_range

# Pile Length Range Slider
length_range = st.sidebar.slider(
    "Length Range (m)",
    min_value=2,
    max_value=100,
    value=(st.session_state.length_min, st.session_state.length_max),
    step=2,
)
st.session_state.length_min, st.session_state.length_max = length_range

# Geotechnical Reduction Factor
st.session_state.reduction_factor = st.sidebar.number_input(
    "Geotechnical Reduction Factor (0 to 1)",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.reduction_factor,
    step=0.05,
)

# Toggles
st.session_state.three_d_embed = st.sidebar.checkbox(
    "3D embedment", value=st.session_state.three_d_embed
)
st.session_state.skin_friction_only = st.sidebar.checkbox(
    "Skin friction only", value=st.session_state.skin_friction_only
)

# --- Geological Units Section ---
st.sidebar.subheader("Geological Units")

remove_unit_index = None
for i, unit in enumerate(st.session_state.units):
    with st.sidebar.expander(f"{unit['name']}", expanded=True):
        # Unit Name
        unit_name = st.text_input(
            f"Name of Unit {i+1}", value=unit["name"], key=f"name_{i}"
        )
        # Top Depth
        top_depth = st.number_input(
            f"Top Depth (m) - {unit_name}",
            min_value=0.0,
            value=unit["top_depth"],
            step=0.1,
            key=f"top_depth_{i}",
        )
        # Skin Friction
        skin_friction = st.number_input(
            f"Skin Friction (kPa) - {unit_name}",
            min_value=0.0,
            value=unit["skin_friction"],
            step=1.0,
            key=f"skin_friction_{i}",
        )
        # End Bearing
        end_bearing = st.number_input(
            f"End Bearing (kPa) - {unit_name}",
            min_value=0.0,
            value=unit["end_bearing"],
            step=1.0,
            key=f"end_bearing_{i}",
        )

        # Update the unit in session_state
        st.session_state.units[i]["name"] = unit_name
        st.session_state.units[i]["top_depth"] = top_depth
        st.session_state.units[i]["skin_friction"] = skin_friction
        st.session_state.units[i]["end_bearing"] = end_bearing

        # Remove Unit Button
        if len(st.session_state.units) > 1:
            if st.button(f"Remove {unit_name}", key=f"remove_unit_{i}"):
                remove_unit_index = i

# Handle Remove Unit
if remove_unit_index is not None:
    st.session_state.units.pop(remove_unit_index)
    rename_units()

# Add Unit Button
if st.sidebar.button("Add Unit"):
    last_top = st.session_state.units[-1]["top_depth"]
    st.session_state.units.append(
        {
            "name": f"Unit {len(st.session_state.units)+1}",
            "top_depth": last_top + 10.0,
            "skin_friction": 0.0,
            "end_bearing": 0.0,
        }
    )
    rename_units()

# --- Main Content ---
st.title("Pile Capacity Calculator")

# Extract parameters
diameters = np.arange(
    st.session_state.diameter_min, st.session_state.diameter_max + 0.0001, 0.3
)
lengths = np.arange(
    st.session_state.length_min, st.session_state.length_max + 0.0001, 2
)

units = st.session_state.units

# Generate the plot
with st.spinner("Calculating..."):
    time.sleep(1)  # Simulate processing delay
    fig = go.Figure()

    for d in diameters:
        capacities = []
        for L in lengths:
            cap = calculate_capacity(
                diameter=d,
                length=L,
                units=units,
                reduction_factor=st.session_state.reduction_factor,
                three_d_embed=st.session_state.three_d_embed,
                skin_friction_only=st.session_state.skin_friction_only,
            )
            capacities.append(cap)

        fig.add_trace(
            go.Scatter(
                x=capacities, y=lengths, mode="lines+markers", name=f"D={d:.1f}m"
            )
        )

    # Invert the y-axis and increase height
    fig.update_yaxes(autorange="reversed", title_text="Pile Length (m)")
    fig.update_xaxes(title_text="Total Capacity (kN)")
    fig.update_layout(
        title="Pile Capacity vs Pile Length",
        legend_title_text="Diameter (m)",
        hovermode="x unified",
        template="plotly_white",
        height=1000,  # Double the typical height
    )

    st.plotly_chart(fig, use_container_width=True)

st.success("Calculation complete.")
