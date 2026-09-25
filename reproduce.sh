#!/usr/bin/env bash
set -euo pipefail

rastpo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rastpo_python=python
data_dir="$rastpo_root/data/processed/window"
output="$rastpo_root/outputs/paper"
device=auto
threads=2

# Main-table settings. Additional experiments are available through run.py.
# Skip only S&P 500 SD-relaxation because of its computational cost.
baseline_seeds=({0..31})
correction_seeds=({0..15})
faces=8
sdp_markets=(ESTX50 FTSE100 KOSPI200 NIKKEI225)

fail() { printf '%s\n' "$*" >&2; exit 2; }
(($# == 0)) || fail "reproduce.sh runs the main-table experiments without arguments. For individual experiments, use: python run.py --help"
[[ -f $data_dir/dataset.json ]] || fail "Prepared data missing. Run: $rastpo_python \"$rastpo_root/data/prepare.py\" --representation window"

execution=(--device "$device" --threads "$threads")
training=(--data "$data_dir" --cardinality 10 --hidden 64 32 --dropout .1
          --epochs 40 --patience 8 --anchor-batch-size 64 --resident-inputs "${execution[@]}")
allocation=(--data "$data_dir" --cardinality 10 --ordering stored --sdp-max-seconds 600
            --ranking-cache "$output/rankings/window" "${execution[@]}")
nominal=(--aggregation mean --uncertainty none --credibility none)

# Record each run.py command, its output, and its completion status.
run() {
    local name=$1
    shift
    local -a command=("$rastpo_python" "$rastpo_root/run.py" "$@")
    mkdir -p -- "$output/logs"
    printf '%q ' "${command[@]}" > "$output/logs/$name.command"
    printf '\n' >> "$output/logs/$name.command"
    printf 'running\n' > "$output/logs/$name.status"
    printf '[%s]\n' "$name"
    if env OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" \
        OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads" \
        "${command[@]}" 2>&1 | tee -a "$output/logs/$name.log"; then
        printf 'complete\n' > "$output/logs/$name.status"
    else
        printf 'failed\n' > "$output/logs/$name.status"
        fail "$name failed; see $output/logs/$name.log"
    fi
}

# Train 32 PFL members, 32 DF-STPO members, and 16 RA-STPO correctors.
# The first 16 PFL members also serve as the frozen RA-STPO anchors.
run train_window_pfl train "${training[@]}" --objective prediction \
    --seeds "${baseline_seeds[@]}" --batch-size 64 --output "$output/models/window/pfl"
run train_window_dfstpo train "${training[@]}" --objective soft \
    --seeds "${baseline_seeds[@]}" --batch-size 64 --output "$output/models/window/dfstpo"
run "train_window_faces_$faces" train "${training[@]}" --objective hard \
    --seeds "${correction_seeds[@]}" --batch-size 128 --faces "$faces" \
    --anchor-cache "$output/models/window/pfl" --output "$output/models/window/faces_$faces"

# Evaluate the ten methods in the main table.
for optimizer in oscar sdp pga afba; do
    market_options=()
    if [[ $optimizer == sdp ]]; then
        market_options=(--markets "${sdp_markets[@]}")
    fi
    run "main_window_historic_$optimizer" evaluate "${allocation[@]}" "${nominal[@]}" \
        --optimizer "$optimizer" "${market_options[@]}" \
        --output "$output/evaluations/window/main/historic_$optimizer"
done
for optimizer in oscar sdp pga afba; do
    market_options=()
    if [[ $optimizer == sdp ]]; then
        market_options=(--markets "${sdp_markets[@]}")
    fi
    run "main_window_pfl_$optimizer" evaluate "${allocation[@]}" "${nominal[@]}" \
        --optimizer "$optimizer" "${market_options[@]}" --forecasts "$output/models/window/pfl" \
        --seeds "${baseline_seeds[@]}" --output "$output/evaluations/window/main/pfl_$optimizer"
done
run main_window_dfstpo evaluate "${allocation[@]}" "${nominal[@]}" --optimizer dfstpo \
    --forecasts "$output/models/window/dfstpo" --seeds "${baseline_seeds[@]}" \
    --output "$output/evaluations/window/main/dfstpo"
run main_window_rastpo evaluate "${allocation[@]}" --optimizer alpha \
    --forecasts "$output/models/window/faces_$faces" --seeds "${correction_seeds[@]}" \
    --aggregation calibrated --uncertainty mean --credibility sharpe \
    --output "$output/evaluations/window/main/rastpo"

# Generate tables from the saved results without further experiment runs.
run summarize summarize --data "$data_dir" --output "$output" \
    --members "${#correction_seeds[@]}" --baseline-members "${#baseline_seeds[@]}" --faces "$faces"

cat >> "$output/RESULTS.md" <<'NOTE'

`reproduce.sh` skips Historic / SD-relaxation and PFL / SD-relaxation on S&P 500
because of computational cost. Missing entries are shown as dashes; any previously
saved results are retained. These cases and additional experiments can be run
individually through `run.py`.
NOTE
