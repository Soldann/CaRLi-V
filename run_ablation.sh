#!/bin/bash
# =============================================================================
# CaRLi-V Ablation Study Runner
# =============================================================================
# Usage: Uncomment ONE line below and run this script.
#        Each line defines a parameter combination and a descriptive save name.
#        The script will launch the full pipeline, play the rosbag, wait for it
#        to finish, then shut down.
#
# Parameters:
#   IT  = intensity_threshold_offset  (default: 5.0)
#   MFE = morph_filter_elev           (default: 10)
#   MFA = morph_filter_azimuth        (default: 10)
#   MFR = morph_filter_range          (default: 20)
#   AZB = target_azimuth_bins         (default: 50)
#   ELB = target_elevation_bins       (default: 2)
#
# After running, PCDs will be saved under datasets/<SAVE_NAME>/predicted/
# =============================================================================

# After installing NeuFlow, set your PYTHONPATH to include the NeuFlow directory so the optical flow node can find it:
# export PYTHONPATH=$HOME/NeuFlow_v2:$PYTHONPATH (add to bashrc for convenience)

# Run this script with the path to your rosbag as an argument, e.g.:
# pkill -f carli_v  # stop any existing runs to free the GPU
# ./run_ablation.sh /mnt/m/carli_v/scene_1.mcap 0.01

# Path to your rosbag (EDIT THIS)
ROSBAG_PATH="${1:?Usage: $0 <path_to_rosbag> [playback_rate]}"
# Playback rate (default 1.0 = real speed; slower rates hurt optical flow quality because
# NeuFlow processes frame pairs that are closer in time, reducing pixel displacement SNR)
PLAYBACK_RATE="${2:-1.0}"

# Get bag duration in seconds from ros2 bag info
get_bag_duration() {
    local info
    info=$(ros2 bag info "$ROSBAG_PATH" 2>/dev/null)
    # Extract duration line like "Duration: 123.456s" or "Duration:    42s"
    echo "$info" | grep -oP 'Duration:\s+\K[0-9]+(\.[0-9]+)?' | head -1
}

BAG_DURATION=$(get_bag_duration)
if [ -z "$BAG_DURATION" ]; then
    echo "Warning: Could not determine bag duration. Progress bar disabled."
fi

# Print a message above the in-progress bar (preserves the bar on its own line)
print_above_bar() {
    printf "\r%-100s\r" " "   # clear current progress bar line
    printf "  %s\n" "$1"      # print the message on its own line
    # progress bar will redraw itself on the next tick
}

# Progress bar that tracks wall-clock time and surfaces important log events
show_progress() {
    local save_name="$1"
    local bag_pid="$2"
    local log_file="$3"
    local wall_duration
    wall_duration=$(awk "BEGIN {printf \"%.1f\", $BAG_DURATION / $PLAYBACK_RATE}")
    local bar_width=36
    local start_time=$SECONDS

    # Temp files so the background tail process can pass counts back to us
    local of_miss_file img_miss_file err_file
    of_miss_file=$(mktemp); img_miss_file=$(mktemp); err_file=$(mktemp)
    echo 0 > "$of_miss_file"; echo 0 > "$img_miss_file"

    # Background log monitor: tally known warnings and collect unexpected errors
    tail -f "$log_file" 2>/dev/null | while IFS= read -r line; do
        case "$line" in
            *"No suitable delayed OPTICAL FLOW"*)
                echo $(( $(cat "$of_miss_file") + 1 )) > "$of_miss_file" ;;
            *"No suitable delayed IMAGE"*)
                echo $(( $(cat "$img_miss_file") + 1 )) > "$img_miss_file" ;;
            *"Error"*|*"Traceback"*|*"Exception"*)
                # Skip overly chatty lines that aren't real errors
                echo "$line" | grep -qvE "rclpy|ParameterNotDeclaredException|INFO" \
                    && echo "$line" >> "$err_file" ;;
        esac
    done &
    local tail_pid=$!

    local last_err_count=0
    local last_of_miss=0 last_img_miss=0

    while kill -0 "$bag_pid" 2>/dev/null; do
        # Surface any new errors immediately above the bar
        local err_count
        err_count=$(wc -l < "$err_file" 2>/dev/null | tr -d ' ')
        if [ "${err_count:-0}" -gt "$last_err_count" ]; then
            tail -n +$(( last_err_count + 1 )) "$err_file" 2>/dev/null | while IFS= read -r errline; do
                print_above_bar "[ERROR] $errline"
            done
            last_err_count=${err_count:-0}
        fi

        # Warn once when OF-miss count first crosses threshold (startup misses are normal)
        local of_miss img_miss
        of_miss=$(cat "$of_miss_file" 2>/dev/null || echo 0)
        img_miss=$(cat "$img_miss_file" 2>/dev/null || echo 0)
        if [ "$of_miss" -ge 10 ] && [ "$last_of_miss" -lt 10 ]; then
            print_above_bar "[WARN] 10+ optical flow misses — pipeline may be falling behind"
        fi
        if [ "$img_miss" -ge 10 ] && [ "$last_img_miss" -lt 10 ]; then
            print_above_bar "[WARN] 10+ image misses — check camera topic timing"
        fi
        last_of_miss=$of_miss; last_img_miss=$img_miss

        # Draw progress bar
        local elapsed=$(( SECONDS - start_time ))
        local pct
        pct=$(awk "BEGIN {p=100*$elapsed/$wall_duration; if(p>100) p=100; printf \"%.0f\", p}")
        local filled=$(( pct * bar_width / 100 ))
        local empty=$(( bar_width - filled ))
        local elapsed_m=$(( elapsed / 60 )) elapsed_s=$(( elapsed % 60 ))
        local total_m=$(( ${wall_duration%.*} / 60 )) total_s=$(( ${wall_duration%.*} % 60 ))

        local warn_str=""
        [ "$of_miss" -gt 0 ]  && warn_str=" | OF-miss:${of_miss}"
        [ "$img_miss" -gt 0 ] && warn_str="${warn_str} img-miss:${img_miss}"

        printf "\r  [%s] [%s] %s%s %3d%%  %02d:%02d/%02d:%02d%s" \
            "$(date +%H:%M:%S)" "$save_name" \
            "$(printf '#%.0s' $(seq 1 $filled 2>/dev/null))" \
            "$(printf -- '-%.0s' $(seq 1 $empty 2>/dev/null))" \
            "$pct" "$elapsed_m" "$elapsed_s" "$total_m" "$total_s" \
            "$warn_str"

        sleep 1
    done

    kill $tail_pid 2>/dev/null
    wait $tail_pid 2>/dev/null
    rm -f "$of_miss_file" "$img_miss_file" "$err_file"

    printf "\r  [%s] [%s] %s 100%%  done%-40s\n" \
        "$(date +%H:%M:%S)" "$save_name" \
        "$(printf '#%.0s' $(seq 1 $bar_width))" ""
}

run_combination() {
    local SAVE_NAME="$1"
    local IT="$2"
    local MFE="$3"
    local MFA="$4"
    local MFR="$5"
    local AZB="$6"
    local ELB="$7"

    echo ""
    echo "============================================="
    echo " $SAVE_NAME"
    echo "   IT=$IT  MF=($MFE,$MFA,$MFR)  AZB=$AZB  ELB=$ELB"
    echo "============================================="

    local LOG_FILE="ablation_${SAVE_NAME}.log"
    echo "  (ROS output -> $LOG_FILE)"

    # Launch the pipeline in background, log output for debugging
    ros2 launch carli_v radar_full_velocity.launch.py \
        save_pcd_as:="$SAVE_NAME" \
        intensity_threshold_offset:="$IT" \
        morph_filter_elev:="$MFE" \
        morph_filter_azimuth:="$MFA" \
        morph_filter_range:="$MFR" \
        target_azimuth_bins:="$AZB" \
        target_elevation_bins:="$ELB" > "$LOG_FILE" 2>&1 &
    LAUNCH_PID=$!

    # Wait for all nodes to be ready — optical_flow_node loads NeuFlow which can take 30+ seconds.
    # Poll until the optical flow publisher is active, with a timeout and live log feedback.
    local wait_start=$SECONDS
    local wait_timeout=180  # seconds before giving up
    printf "  Waiting for optical_flow_node to be ready (timeout: ${wait_timeout}s)\n"
    while ! ros2 topic info /optical_flow_uv_map 2>/dev/null | grep -q "Publisher count: 1"; do
        local waited=$(( SECONDS - wait_start ))
        # Check if the launch process died (node crashed during startup)
        if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
            printf "\n  [FAILED] Launch process died. Check log for errors:\n"
            grep -i "error\|cuda\|traceback\|exception\|RuntimeError" "$LOG_FILE" 2>/dev/null | tail -10 | sed 's/^/    /'
            printf "  Full log: %s\n" "$LOG_FILE"
            return 1
        fi
        # Show last relevant log line so the user can see what's happening
        local last_line
        last_line=$(grep -v "^\s*$" "$LOG_FILE" 2>/dev/null | tail -1)
        printf "\r  [%3ds] %s" "$waited" "${last_line:0:90}"
        if [ "$waited" -ge "$wait_timeout" ]; then
            printf "\n  [TIMEOUT] optical_flow_node did not start after ${wait_timeout}s.\n"
            printf "  Last log lines:\n"
            tail -10 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
            printf "  Full log: %s\n" "$LOG_FILE"
            kill $LAUNCH_PID 2>/dev/null
            return 1
        fi
        sleep 2
    done
    printf "\r%-100s\r" " "
    printf "  optical_flow_node ready (after $(( SECONDS - wait_start ))s)\n"

    # Play the rosbag at reduced speed, log output for debugging
    ros2 bag play "$ROSBAG_PATH" --clock --rate "$PLAYBACK_RATE" >> "$LOG_FILE" 2>&1 &
    BAG_PID=$!

    # Show progress bar if we know the duration
    if [ -n "$BAG_DURATION" ]; then
        show_progress "$SAVE_NAME" "$BAG_PID" "$LOG_FILE"
    fi

    wait $BAG_PID 2>/dev/null

    sleep 5  # allow final messages to be processed

    # Kill the launch and ALL child processes (ros2 launch spawns separate node processes)
    kill $LAUNCH_PID 2>/dev/null
    pkill -P $LAUNCH_PID 2>/dev/null  # kill children of the launch process
    wait $LAUNCH_PID 2>/dev/null
    # Ensure no lingering node processes hold the GPU for the next run
    pkill -f "optical_flow_node" 2>/dev/null
    pkill -f "radar_cube_node" 2>/dev/null
    pkill -f "radar_full_velocity_node" 2>/dev/null
    sleep 2  # give GPU time to release

    local saved=$(ls "$(pwd)/datasets/$SAVE_NAME/predicted/" 2>/dev/null | wc -l)
    local warn=$(grep -c "No suitable delayed" "$LOG_FILE" 2>/dev/null || echo "?")
    echo "  Done: $SAVE_NAME -> $saved PCDs saved, $warn skipped frames (see $LOG_FILE)"
}

# =============================================================================
# PARAMETER COMBINATIONS
# Uncomment the combinations you want to run.
# Format: run_combination <save_name> <IT> <MFE> <MFA> <MFR> <AZB> <ELB>
# =============================================================================

# --- Baseline (default parameters) ---
# run_combination "baseline"                  5.0  10 10 20  50 2

# --- Intensity Threshold Offset sweep ---
# run_combination "it_3"                      3.0  10 10 20  50 2
# run_combination "it_7"                      7.0  10 10 20  50 2
# run_combination "it_10"                     10.0 10 10 20  50 2
run_combination "it_13"                     13.0 10 10 20  50 2
run_combination "it_16"                     16.0 10 10 20  50 2
# run_combination "it_20"                     20.0 10 10 20  50 2

# --- Morphological Filter Size sweep ---
# run_combination "mf_5_5_10"                 5.0  5  5  10  50 2
# run_combination "mf_5_5_20"                 5.0  5  5  20  50 2
# run_combination "mf_10_10_10"               5.0  10 10 10  50 2
# run_combination "mf_15_15_30"               5.0  15 15 30  50 2
# run_combination "mf_20_20_40"               5.0  20 20 40  50 2
# run_combination "mf_5_10_20"                5.0  5  10 20  50 2

# --- Target Azimuth Bins sweep ---
#run_combination "azb_25"                    5.0  10 10 20  25 2
#run_combination "azb_75"                    5.0  10 10 20  75 2
#run_combination "azb_100"                   5.0  10 10 20  100 2

# --- Target Elevation Bins sweep ---
#run_combination "elb_4"                     5.0  10 10 20  50 4
#run_combination "elb_8"                     5.0  10 10 20  50 8
#run_combination "elb_16"                    5.0  10 10 20  50 16

# --- Combined azimuth + elevation ---
#run_combination "azb_25_elb_4"              5.0  10 10 20  25 4
#run_combination "azb_75_elb_4"              5.0  10 10 20  75 4
#run_combination "azb_100_elb_8"             5.0  10 10 20  100 8
