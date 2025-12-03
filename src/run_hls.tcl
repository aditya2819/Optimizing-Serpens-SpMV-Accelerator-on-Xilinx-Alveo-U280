##########################################
# Vitis HLS Script for Serpens (pure HLS)
##########################################

open_project serpens_hls_prj

# Clean solution if exists
if { [file exists serpens_hls_prj/sol1] } {
    close_project
    file delete -force serpens_hls_prj
    open_project serpens_hls_prj
}

# -------------------------------
# Set top function
# -------------------------------
set_top Serpens

# -------------------------------
# Add your renamed source files
# -------------------------------
add_files serpens.cpp
add_files serpens.h

# Testbench (optional, host-side driver)
add_files -tb serpens-host.cpp

# -------------------------------
# Create solution
# -------------------------------
open_solution "sol1"

# U280 device
set_part {xcu280-fsvh2892-2L-e}

# Kernel clock target
create_clock -period 3.2 -name default

# Export config must be set before synthesis
config_export -format xo -rtl verilog

# -------------------------------
# C Simulation (optional)
# -------------------------------
# csim_design

# -------------------------------
# C Synthesis
# -------------------------------
csynth_design

# -------------------------------
# Export XO for v++
# -------------------------------
export_design -output "Serpens.xo"

exit
