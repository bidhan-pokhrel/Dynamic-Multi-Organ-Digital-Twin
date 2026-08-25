"""
app.py
======
Streamlit front end for the whole-body digital twin.

25 Aug 2026 rebuild. Two, deliberately different, data sources now feed this
app instead of one:

  COUPLED / multi-organ tabs (Run simulation, Coupling ablation, Parameter
  sweep, Disease scenarios, Drug interventions, Coupling graph) run through
  `model_artifact.pkl` (from `export_model.py`) and `HumanBody`
  (`integration/human_body.py`), in HYBRID mode by default -- kidney, liver,
  lungs and pancreas advanced by their trained BiLSTM surrogates
  (`bilstm_models/`) rather than solved as ODEs each step, exactly as
  `HumanBody(surrogate_organs=...)` was wired to do on 24 Aug 2026. A
  sidebar toggle switches any of these tabs back to pure ODE physics.

  SINGLE-ORGAN tab runs through `organ_models/<organ>_artifact.pkl` (from
  the new `export_organ_models.py`) and `organ_runner.py` -- one organ,
  alone, always pure ODE physics (no BiLSTM anywhere in this tab; see
  `organ_runner.py`'s module docstring for why an isolated organ needs no
  surrogate at all).

This app never re-reads human_body.py's PARAMETERS or ode_models/*.py
directly -- both artifact types are frozen, pickled snapshots, the same
"config in the pkl, code imported separately" pattern this project has used
since export_model.py. It still needs the actual HumanBody / organ classes
importable to run anything, resolved at run time exactly the way the
notebooks do it.

IMPORTANT CAVEAT ABOUT HYBRID MODE, READ BEFORE TRUSTING AN ABLATION OR
SWEEP RESULT: the trained surrogates were each fit on a single organ's own
ISOLATED synthetic data (WBS 3.x, before coupling existed), so a surrogate's
inputs are exactly that organ's own state-and-input columns -- nothing more.
Three of the nine coupling edges in COUPLING_MAP feed a surrogate-driven
organ a quantity its surrogate was never trained to see at all, so changing
that edge's gain has NO effect on that organ's hybrid-mode output:

  - glucose_to_liver_gain (pancreas -> liver): the liver surrogate reads
    the pancreas's raw plasma glucose, not this gain's scaled version.
  - hepatic_flux_to_glucose_gain (liver -> pancreas): the pancreas
    surrogate's only external input is the gut meal; it never sees hepatic
    flux at all.
  - hepatic_vo2_gain (liver -> lungs): the lungs surrogate's inputs don't
    include an oxygen-consumption term.

Disease scenarios and drug interventions go further still: both work by
changing an organ's ODE *parameters* (RVR, HGP_max, gamma, ...), and a
surrogate's prediction depends only on its trained weights and its own
recent trajectory -- never on those parameters. A CKD preset raising kidney
RVR, or a drug nudging it, would leave a surrogate-driven kidney's predicted
plasma volume and BUN completely unmoved, silently breaking the very
cross-organ effect each scenario/drug exists to demonstrate. For that
reason those two tabs always run pure physics, unconditionally -- see their
own captions below for why, in place, rather than only in this docstring.

Run:
    streamlit run app.py

If model_artifact.pkl or organ_models/*.pkl are missing, or you've changed
human_body.py / ode_models/*.py, run `python export_model.py` and
`python export_organ_models.py` in this folder first, then reload. See
README.md for details.
"""

import os
import sys
import copy
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# Make HumanBody and the organ ODE classes importable. The pickled artifacts
# are plain data; only the *runners* need the actual code, resolved the same
# way human_body.py itself resolves ode_models (relative to this file).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_INTEGRATION_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "integration"))
_SCENARIOS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "scenarios"))
_INTERVENTIONS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "interventions"))
for _dir in (_INTEGRATION_DIR, _SCENARIOS_DIR, _INTERVENTIONS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from human_body import HumanBody  # noqa: E402
from disease_scenarios import (  # noqa: E402
    SCENARIOS, run_scenario, evaluate_effects,
    organ_impact_summary as disease_organ_impact_summary,
    ORGAN_SIGNATURE_COLUMNS as DISEASE_ORGAN_SIGNATURE_COLUMNS,
)
from drug_interventions import (  # noqa: E402
    DRUGS, ORGAN_DRUGS, ORGAN_LABELS, ORGAN_SIGNATURE_COLUMNS,
    run_intervention, evaluate_intervention, organ_impact_summary, peak_time,
    run_multi_intervention, organ_impact_summary_multidrug,
)
import organ_runner  # noqa: E402
# Note: streamlit_app/reference_medications.py (a static, non-simulated
# pharmacology reference table) still exists and is still tested by
# tests/test_reference_medications.py, but is deliberately not imported or
# rendered here -- the app no longer shows it. Kept on disk rather than
# deleted so it's a one-line change to bring back, not a file recovery.

ARTIFACT_PATH = os.path.join(_THIS_DIR, "model_artifact.pkl")
ORGAN_MODELS_DIR = os.path.join(_THIS_DIR, "organ_models")

st.set_page_config(page_title="Coupled Human Body — Digital Twin", layout="wide")

# A fixed, distinct colour per variable, held constant everywhere it appears
# so a colour always means the same thing across tabs.
COLORS = {
    "MAP_mmHg": "#c0392b",
    "Cardiac_Output_L_min": "#e67e22",
    "Stroke_Volume_mL": "#d35400",
    "Heart_Rate_BPM": "#e74c3c",
    "PaO2_mmHg": "#2e86ab",
    "PaCO2_mmHg": "#8e44ad",
    "SaO2_percent": "#1b7f9e",
    "Glucose_mg_dL": "#e08214",
    "Insulin_uU_mL": "#b8860b",
    "Glycogen_Fill_Fraction": "#2f9e44",
    "Plasma_Volume_L": "#2c5282",
    "BUN_mg_dL": "#a0522d",
    "GFR_L_min": "#0f766e",
    "Altitude_m": "#6b7280",
    "Arterial_Pressure_mmHg": "#c0392b",
    "Blood_Flow_mL_s": "#e67e22",
    "Glycogen_Mass_mg": "#2f9e44",
    "HGP_mg_min": "#2f9e44",
    "HGU_mg_min": "#a0522d",
    "Urine_Output_L_min": "#2c5282",
    "Insulin_Action_1_min": "#b8860b",
}

# Human-readable, unit-labeled y-axis labels for every column this app ever
# plots -- shown instead of the raw DataFrame column name (e.g. "GFR_L_min")
# so a chart reads on its own without the reader having to decode the
# variable name. Falls back to the raw column name for anything not listed
# here, so a new column never crashes a plot, just looks less polished.
AXIS_LABELS = {
    "MAP_mmHg": "Mean arterial pressure (mmHg)",
    "Cardiac_Output_L_min": "Cardiac output (L/min)",
    "Stroke_Volume_mL": "Stroke volume (mL)",
    "Heart_Rate_BPM": "Heart rate (bpm)",
    "PaO2_mmHg": "Arterial oxygen, PaO2 (mmHg)",
    "PaCO2_mmHg": "Arterial CO2, PaCO2 (mmHg)",
    "SaO2_percent": "Oxygen saturation, SaO2 (%)",
    "Glucose_mg_dL": "Blood glucose (mg/dL)",
    "Insulin_uU_mL": "Plasma insulin (uU/mL)",
    "Glycogen_Fill_Fraction": "Liver glycogen store (fraction full)",
    "Glycogen_Mass_mg": "Liver glycogen mass (mg)",
    "HGP_mg_min": "Hepatic glucose production (mg/min)",
    "HGU_mg_min": "Hepatic glucose uptake (mg/min)",
    "Plasma_Volume_L": "Blood plasma volume (L)",
    "BUN_mg_dL": "Blood urea nitrogen, BUN (mg/dL)",
    "GFR_L_min": "Kidney filtration rate, GFR (L/min)",
    "Urine_Output_L_min": "Urine output (L/min)",
    "Altitude_m": "Altitude (m)",
    "Arterial_Pressure_mmHg": "Arterial pressure (mmHg)",
    "Blood_Flow_mL_s": "Blood flow (mL/s)",
    "Insulin_Action_1_min": "Active insulin action (1/min)",
}


def _axis_label(col):
    """Human-readable, unit-labeled y-axis label for a DataFrame column --
    falls back to the raw column name if it isn't in AXIS_LABELS."""
    return AXIS_LABELS.get(col, col)

# The organs a hybrid-mode run advances with their trained BiLSTM surrogates
# instead of solving their ODEs -- see integration/surrogates.py. Heart is
# never in this set (different timescale, see that module's docstring).
HYBRID_ORGANS = ("kidney", "liver", "lungs", "pancreas")
# The surrogates' own training cadence (integration/surrogates.py /
# bilstm_models/<organ>_bundle_portable.json all agree: 5 minutes). Hybrid
# mode requires the whole body to step at exactly this dt -- HumanBody.step
# raises a clear error otherwise.
HYBRID_DT = 5.0

# The three coupling edges a surrogate-driven organ cannot see at all -- see
# this module's own docstring for the code-level reason each one is inert.
# {edge_name: (organ whose surrogate never receives it, one-line why)}
INERT_UNDER_HYBRID = {
    "glucose_to_liver_gain": (
        "liver",
        "the liver surrogate reads the pancreas's raw plasma glucose "
        "(Blood_Glucose_mg_dL), not this gain's scaled value"),
    "hepatic_flux_to_glucose_gain": (
        "pancreas",
        "the pancreas surrogate's only external input is the gut meal "
        "(Meal_Ra_mg_min); it never receives hepatic flux at all"),
    "hepatic_vo2_gain": (
        "lungs",
        "the lungs surrogate's inputs (arterial gases, inspired O2, "
        "ventilation, diffusing capacity) don't include an oxygen-"
        "consumption term"),
}

_HYBRID_ORGAN_CAVEAT = (
    "Under hybrid mode, **%s**'s plotted state is produced by its trained "
    "BiLSTM surrogate every step, not by integrating these ODE parameters "
    "-- so changing %s's own parameters here may show little to no effect "
    "on %s's state variables themselves. A few outputs are still computed "
    "algebraically from these parameters every step regardless of which "
    "path is advancing the state (for example GFR_L_min for the kidney, or "
    "HGP/HGU/Hepatic_VO2 for the liver) and can still move -- if a result "
    "looks flat, compare it against a Pure-physics run before concluding "
    "the parameter has no effect."
)

# ---------------------------------------------------------------------------
# Plain-language glosses for otherwise-technical dropdown items, shown in
# brackets next to the parameter's real (code) name via st.selectbox's
# format_func -- the underlying value passed around the app is always still
# the technical key, this only changes what's printed on screen. Kept as
# small UI-only dicts here rather than edited into export_model.py's
# PARAM_SCHEMA, so adding/adjusting a gloss never requires re-exporting and
# redelivering model_artifact.pkl.
# ---------------------------------------------------------------------------
_COUPLING_PLAIN_LABELS = {
    "preload_gain": "kidney fluid level raising heart filling",
    "hypoxia_sv_gain": "low blood oxygen weakening heart contractions",
    "hypoxia_ref_PaO2": "oxygen level below which that weakening starts",
    "sympathetic_hr_gain": "low blood oxygen raising heart rate",
    "sympathetic_ref_PaO2": "oxygen level below which that heart-rate rise starts",
    "co2_vasodilation_gain": "high blood CO2 widening blood vessels",
    "co2_ref_PaCO2": "CO2 level above which that vessel-widening starts",
    "perfusion_dl_exponent": "heart output changing how well lungs transfer oxygen",
    "co_ref": "reference heart output for that lung-transfer effect",
    "map_to_kidney_gain": "how much blood pressure the kidney actually feels",
    "glucose_to_liver_gain": "blood sugar signal reaching the liver",
    "hepatic_flux_to_glucose_gain": "liver's glucose output feeding back into blood sugar",
    "hepatic_vo2_gain": "liver's oxygen use adding to the body's total demand",
}

_LUNGS_PLAIN_LABELS = {
    "PiO2": "inspired oxygen level (room air)",
    "VA": "breathing rate/depth (alveolar ventilation)",
    "DL": "how easily oxygen crosses into the blood",
    "VO2": "oxygen used by the body",
    "VCO2": "carbon dioxide produced by the body",
    "RQ": "CO2 produced per oxygen used",
    "Vcap": "blood volume in the lungs' small vessels",
    "hvr_gain": "how strongly low oxygen speeds up breathing",
    "hvr_threshold": "oxygen level that triggers faster breathing",
    "hvr_max": "cap on how much faster low oxygen can make you breathe",
    "recruit_gain": "how much extra lung blood vessels open under stress",
    "P50": "how tightly blood holds onto oxygen",
    "hill_n": "steepness of the blood-oxygen binding curve",
    "exercise_factor": "how much harder the body works during exercise",
    "altitude_target": "peak altitude reached",
    "altitude_after": "altitude held after descending",
}
# Timing knobs for a scripted exercise/altitude profile -- rarely what
# someone sweeping "lungs" is actually trying to explore, and (being start/
# end minute pairs) awkward to sweep meaningfully on their own; dropped from
# the Parameter dropdown for lungs specifically, per explicit request.
_LUNGS_SWEEP_EXCLUDE = {
    "exercise_start", "exercise_end",
    "ascent_start", "ascent_end", "descent_start", "descent_end",
}

_KIDNEY_PLAIN_LABELS = {
    "RVR": "resistance to blood flow in the kidney",
    "FF": "share of plasma the kidney filters out",
    "Hct": "haematocrit (red blood cell percentage)",
    "f_reab": "fraction of filtered fluid reclaimed back into the blood",
    "Vp_ref": "target (set-point) blood plasma volume",
    "U_prod": "urea production rate",
    "Vp_initial": "starting blood plasma volume",
    "BUN_initial": "starting blood urea nitrogen (kidney-function marker)",
    "drink_start": "when fluid intake begins",
    "drink_end": "when fluid intake ends",
    "drink_rate": "fluid intake rate while drinking",
    "baseline_intake": "background water intake outside the drink window",
}

# Parameter-sweep dropdown: organ -> its plain-language gloss dict. Only
# lungs and kidney were explicitly requested; other organs still show their
# technical parameter names as before.
_SWEEP_PLAIN_LABELS = {
    "lungs": _LUNGS_PLAIN_LABELS,
    "kidney": _KIDNEY_PLAIN_LABELS,
}

# Timing parameters whose full 0-1440 min schema range is almost always
# wider than the sweep's own Duration -- sweeping past Duration produces
# rows where the scripted event (drink/meal) never happens within the run,
# which look flat/uninformative rather than broken. See the sweep tab below
# for how this is used to pick a sane default range and warn when exceeded.
_SWEEP_TIMING_PARAMS = {"drink_start", "drink_end", "meal_start", "meal_end"}


def _sweep_param_label(param_name, plain_labels):
    gloss = plain_labels.get(param_name)
    return "%s (%s)" % (param_name, gloss) if gloss else param_name


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifact():
    if not os.path.exists(ARTIFACT_PATH):
        st.error(
            "No model_artifact.pkl found next to app.py.\n\n"
            "Run `python export_model.py` in this folder first, then reload "
            "this page. See README.md for details."
        )
        st.stop()
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_organ_artifacts():
    organs = ("heart", "lungs", "liver", "kidney", "pancreas")
    missing = [o for o in organs if not os.path.exists(
        os.path.join(ORGAN_MODELS_DIR, "%s_artifact.pkl" % o))]
    if missing:
        st.error(
            "Missing organ artifact(s) for: %s.\n\n"
            "Run `python export_organ_models.py` in this folder first, "
            "then reload this page." % ", ".join(missing)
        )
        st.stop()
    out = {}
    for o in organs:
        with open(os.path.join(ORGAN_MODELS_DIR, "%s_artifact.pkl" % o), "rb") as f:
            out[o] = pickle.load(f)
    return out


artifact = load_artifact()
BASE_PARAMS = artifact["parameters"]
SCHEMA = artifact["param_schema"]
SECTION_ORDER = artifact["section_order"]
COUPLING_MAP = artifact["coupling_map"]
FEEDBACK_LOOPS = artifact["feedback_loops"]

ORGAN_ARTIFACTS = load_organ_artifacts()

if "params" not in st.session_state:
    st.session_state.params = copy.deepcopy(BASE_PARAMS)
if "sim_mode" not in st.session_state:
    st.session_state.sim_mode = "hybrid"   # "hybrid" or "physics"
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "compare_df" not in st.session_state:
    st.session_state.compare_df = None
if "compare_df_key" not in st.session_state:
    # (sweep_section, sweep_param) the current compare_df was computed for.
    # Cleared whenever the Organ/Parameter dropdowns change so a stale table
    # from a *different* selection can never be mistaken for a fresh one --
    # see the sweep tab below.
    st.session_state.compare_df_key = None
if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None
if "drug_result" not in st.session_state:
    st.session_state.drug_result = {}   # keyed by drug name, so switching organ tabs doesn't lose results
if "_pending_resets" not in st.session_state:
    # (section, name) pairs whose per-slider "reset to default" button was
    # just clicked. Streamlit forbids writing to st.session_state[key] for
    # a widget key that has *already* been instantiated earlier in the same
    # script run -- so a reset button can't just poke the slider's own key
    # and rerun, because the slider for that key was already drawn earlier
    # in that same run. Queuing the reset here and applying it at the very
    # start of the *next* run, before that slider is instantiated, avoids
    # the "cannot be modified after the widget ... is instantiated" error.
    st.session_state._pending_resets = set()
if "_pending_overrides" not in st.session_state:
    # {(section, name): forced_value} -- same "queue it, apply it at the
    # very start of the next run, before that widget is instantiated"
    # mechanism as _pending_resets above, used by the start/end validation
    # block below the sidebar loop (its "Swap" buttons need to write to
    # TWO widgets' keys at once -- both members of a start/end pair --
    # which a plain reset-style queue keyed by a single (section, name)
    # can't express).
    st.session_state._pending_overrides = {}
if "organ_params" not in st.session_state:
    # {organ: {**constructor_params, **protocol_params}} -- the single-organ
    # tab's own working copy, separate from st.session_state.params (the
    # coupled body's config) since the two are different artifacts entirely.
    st.session_state.organ_params = {
        o: {**copy.deepcopy(a["constructor_params"]), **copy.deepcopy(a["protocol_params"])}
        for o, a in ORGAN_ARTIFACTS.items()
    }
if "organ_context" not in st.session_state:
    st.session_state.organ_context = {
        o: {k: v["default"] for k, v in a["context_schema"].items()}
        for o, a in ORGAN_ARTIFACTS.items()
    }
if "organ_result" not in st.session_state:
    st.session_state.organ_result = {}   # keyed by organ


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

st.sidebar.title("Configuration")
st.sidebar.caption(
    "Coupled-body artifact exported %s from %s" % (artifact["exported_at"], artifact["source_file"])
)

if st.sidebar.button("Reset all parameters to defaults", use_container_width=True):
    st.session_state.params = copy.deepcopy(BASE_PARAMS)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Coupled simulation mode")
sim_mode_label = st.sidebar.radio(
    "How should the five-organ body be advanced?",
    ["Hybrid — BiLSTM surrogates", "Pure physics — ODE only"],
    index=0 if st.session_state.sim_mode == "hybrid" else 1,
    help=(
        "Hybrid: kidney, liver, lungs and pancreas are advanced by their "
        "trained BiLSTM surrogates once a 2-hour warm-up window fills "
        "(physics runs during warm-up); the heart is always physics, "
        "different timescale. Pure physics: every organ is solved from its "
        "ODE every step, as the model has always done. Applies to Run "
        "simulation, Coupling ablation and Parameter sweep -- Disease "
        "scenarios and Drug interventions always run pure physics "
        "regardless of this toggle (see their own tabs for why)."
    ),
)
st.session_state.sim_mode = "hybrid" if sim_mode_label.startswith("Hybrid") else "physics"
SIM_MODE = st.session_state.sim_mode
if SIM_MODE == "hybrid":
    st.sidebar.caption(
        "dt is fixed at %.0f min (the surrogates' training cadence) while "
        "hybrid mode is on." % HYBRID_DT
    )

st.sidebar.markdown("---")


_TIME_PARAM_NAMES = {
    # The ten "when does this happen" parameters across the four disturbance
    # protocols (meal, drink, exercise, altitude ascent/descent). All ten
    # share the same shape in SCHEMA: unit "min", meaning minutes elapsed
    # since the run starts (t = 0), *not* a duration -- e.g. meal_start=300,
    # meal_end=360 is a meal from the 5-hour mark to the 6-hour mark, a
    # 60-minute meal, not a 360-minute one.
    #
    # Previously rendered as an "elapsed h : min" pair of number inputs;
    # changed to a single plain-minutes number input on 25 Aug 2026 at the
    # student's direct request ("only minutes makes sense"). Still stored
    # as a single minutes float in st.session_state.params either way, so
    # HumanBody itself never saw hours and doesn't change either way.
    "meal_start", "meal_end",
    "drink_start", "drink_end",
    "exercise_start", "exercise_end",
    "ascent_start", "ascent_end",
    "descent_start", "descent_end",
}

# The five (start, end) pairs above, grouped -- used by the start/end
# validation block below the sidebar loop. Added 25 Aug 2026 after the
# student noticed the app silently accepts exercise_end before
# exercise_start: HumanBody's own "pl['exercise_start'] <= t <=
# pl['exercise_end']" check (see human_body.py) can never be true when
# start > end, so the disturbance simply never fires for the rest of the
# run -- no error, no warning, just a flat resting trace. Checked the same
# pattern for the other four pairs directly against human_body.py rather
# than assuming by symmetry: meal_Ra and fluid_intake use the identical
# "start <= t <= end" form (same silent-no-op failure mode), while
# altitude's ascent/descent pair is used in a ramp fraction
# `(t - start) / max(1e-9, end - start)` -- if end < start that
# denominator collapses to the 1e-9 floor instead of a sensible negative
# span, so a swapped ascent or descent pair doesn't go quiet, it produces
# an enormous, unphysical altitude spike instead. Both failure modes are
# worth catching in the UI, and this file does not touch human_body.py or
# ode_models/ to do it -- see the validation block's own comment for why a
# UI-level check is sufficient here.
_START_END_PAIRS = (
    ("pancreas", "meal_start", "meal_end"),
    ("kidney", "drink_start", "drink_end"),
    ("lungs", "exercise_start", "exercise_end"),
    ("lungs", "ascent_start", "ascent_end"),
    ("lungs", "descent_start", "descent_end"),
)


def _render_time_control(section, name, meta, container):
    """Plain-minutes control for the ten parameters in _TIME_PARAM_NAMES
    (meal/drink/exercise/altitude ascent & descent start/end) -- see that
    set's own comment for what these values mean. Same reset-button
    contract as render_param_control's plain slider path below (see its
    own docstring for why resets have to be queued in _pending_resets
    rather than applied immediately), plus the same queued-override
    mechanism (_pending_overrides) the start/end validation block's "Swap"
    buttons use to fix a reversed pair -- both need to write into this
    widget's session_state key *before* it is instantiated this run, which
    is exactly what queue-then-rerun achieves.
    """
    ctrl_key = "ctrl_%s_%s" % (section, name)
    reset_key = "reset_%s_%s" % (section, name)
    default_val = BASE_PARAMS[section][name]

    if (section, name) in st.session_state._pending_resets:
        st.session_state._pending_resets.discard((section, name))
        st.session_state[ctrl_key] = float(default_val)
        st.session_state.params[section][name] = default_val

    if (section, name) in st.session_state._pending_overrides:
        forced = st.session_state._pending_overrides.pop((section, name))
        st.session_state[ctrl_key] = float(forced)
        st.session_state.params[section][name] = float(forced)

    current = float(np.clip(st.session_state.params[section][name], meta["min"], meta["max"]))

    minutes_col, reset_col = container.columns([5, 1])
    val = minutes_col.number_input(
        "%s (min, elapsed since run start)" % name,
        min_value=float(meta["min"]), max_value=float(meta["max"]),
        value=current, step=float(meta["step"]),
        key=ctrl_key, help=meta["help"],
    )
    st.session_state.params[section][name] = val
    is_default = abs(val - float(default_val)) < 1e-9
    reset_col.write("")   # vertical spacer so the button lines up with the number input
    if reset_col.button(
        "↺", key=reset_key, disabled=is_default,
        help=("Already at its default" if is_default else "Reset to default (%.0f min)" % default_val),
    ):
        st.session_state._pending_resets.add((section, name))
        st.rerun()


def render_param_control(section, name, meta, container):
    """One control for one parameter, reading/writing session_state.params.

    Every control gets its own small "reset to default" button next to it,
    separate from the "Reset all parameters to defaults" button above --
    that one discards every slider at once, this one snaps just this one
    parameter back to the healthy value it started at in BASE_PARAMS (the
    values `export_model.py` last exported from human_body.py's
    PARAMETERS), without touching anything else you've already tuned.

    Implementation note: a Streamlit widget with an explicit `key` can only
    have its st.session_state[key] entry written to *before* that widget is
    instantiated in a given script run -- writing to it afterwards (e.g.
    from the reset button's own on-click code, which necessarily runs after
    the slider above it has already been drawn this run) raises
    "cannot be modified after the widget ... is instantiated". So a click
    only *queues* the reset (in st.session_state._pending_resets) and
    reruns; the queued reset is then applied here, before the widget for
    that key is created, on the very next run -- see the pending-reset
    block immediately below.

    The ten meal/drink/exercise/altitude timing parameters are handled
    separately by _render_time_control (a plain-minutes number input, plus
    the queued-override path its "Swap" button in the start/end validation
    block needs) -- see _TIME_PARAM_NAMES. `dt` in the simulation section is
    handled by the caller (see the sidebar loop below) rather than here,
    since whether it's editable at all depends on the hybrid/physics mode
    toggle.
    """
    if name in _TIME_PARAM_NAMES:
        _render_time_control(section, name, meta, container)
        return

    ctrl_key = "ctrl_%s_%s" % (section, name)
    reset_key = "reset_%s_%s" % (section, name)
    pin_key = "pin_%s_%s" % (section, name)
    default_val = BASE_PARAMS[section][name]

    if (section, name) in st.session_state._pending_resets:
        st.session_state._pending_resets.discard((section, name))
        if name == "BUN_initial":
            st.session_state[pin_key] = default_val is not None
            if default_val is not None:
                st.session_state[ctrl_key] = float(default_val)
            elif ctrl_key in st.session_state:
                del st.session_state[ctrl_key]
        else:
            st.session_state[ctrl_key] = float(default_val)
        st.session_state.params[section][name] = default_val

    current = st.session_state.params[section][name]
    label = "%s (%s)" % (name, meta["unit"]) if meta["unit"] != "-" else name

    # BUN_initial is special: None means "solve its own steady state," and
    # that unpinned state is also its healthy default, so resetting it
    # means un-pinning it rather than moving a slider.
    if name == "BUN_initial":
        default_is_pinned = default_val is not None
        slider_col, reset_col = container.columns([5, 1])
        pinned = slider_col.checkbox(
            "Pin starting BUN instead of solving its steady state",
            value=current is not None,
            key=pin_key,
            help=meta["help"],
        )
        if pinned:
            slider_default = current if current is not None else 12.0
            val = slider_col.slider(
                label, min_value=float(meta["min"]), max_value=float(meta["max"]),
                value=float(slider_default), step=float(meta["step"]),
                key=ctrl_key, help=meta["help"],
            )
            st.session_state.params[section][name] = val
        else:
            st.session_state.params[section][name] = None
        reset_col.write("")   # vertical spacer so the button lines up with the checkbox row
        if reset_col.button("↺", key=reset_key, help="Reset to default (%s)" % (
                "pinned at %.4g" % default_val if default_is_pinned else "unpinned -- solves its own steady state")):
            st.session_state._pending_resets.add((section, name))
            st.rerun()
        return

    slider_col, reset_col = container.columns([5, 1])
    val = slider_col.slider(
        label,
        min_value=float(meta["min"]), max_value=float(meta["max"]),
        value=float(np.clip(current, meta["min"], meta["max"])),
        step=float(meta["step"]),
        key=ctrl_key,
        help=meta["help"],
    )
    st.session_state.params[section][name] = val
    is_default = abs(val - float(default_val)) < 1e-9
    reset_col.write("")   # vertical spacer so the button lines up with the slider, not its label
    if reset_col.button(
        "↺", key=reset_key, disabled=is_default,
        help=("Already at its default" if is_default else "Reset to default (%.4g)" % default_val),
    ):
        st.session_state._pending_resets.add((section, name))
        st.rerun()


_SOLVER_OPTIONS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
_SOLVER_CTRL_KEY = "ctrl_simulation_solver"
_SOLVER_RESET_TOKEN = ("simulation", "solver")   # shares the same pending-reset queue as the sliders

if _SOLVER_RESET_TOKEN in st.session_state._pending_resets:
    st.session_state._pending_resets.discard(_SOLVER_RESET_TOKEN)
    _solver_default = BASE_PARAMS["simulation"].get("solver", "RK45")
    st.session_state[_SOLVER_CTRL_KEY] = _solver_default
    st.session_state.params["simulation"]["solver"] = _solver_default

for section, title in SECTION_ORDER:
    with st.sidebar.expander(title, expanded=False):
        for name, meta in SCHEMA[section].items():
            if section == "simulation" and name == "dt":
                if SIM_MODE == "hybrid":
                    st.caption(
                        "dt (min) -- fixed at %.0f while hybrid mode is on. "
                        "Switch to Pure physics in the sidebar above to "
                        "make this adjustable again." % HYBRID_DT
                    )
                    st.session_state.params[section][name] = HYBRID_DT
                    continue
            render_param_control(section, name, meta, st)
        if section == "simulation":
            solver_default = BASE_PARAMS["simulation"].get("solver", "RK45")
            solver_col, solver_reset_col = st.columns([5, 1])
            solver = solver_col.selectbox(
                "solver", options=_SOLVER_OPTIONS,
                index=_SOLVER_OPTIONS.index(
                    st.session_state.params["simulation"].get("solver", "RK45")
                ),
                key=_SOLVER_CTRL_KEY,
                help="scipy integration method used for every organ each step "
                     "(physics organs only, in hybrid mode).",
            )
            st.session_state.params["simulation"]["solver"] = solver
            solver_reset_col.write("")
            if solver_reset_col.button(
                "↺", key="reset_simulation_solver",
                disabled=(solver == solver_default),
                help="Reset to default (%s)" % solver_default,
            ):
                st.session_state._pending_resets.add(_SOLVER_RESET_TOKEN)
                st.rerun()

# ---------------------------------------------------------------------------
# Start/end validation for the five disturbance windows
# ---------------------------------------------------------------------------
# Runs right after the section expanders above, so it reads THIS run's
# freshly-written st.session_state.params (each _render_time_control call
# above already wrote its own widget's value into params by the time we
# get here) rather than stale values left over from the previous run. A
# "Swap" click below queues a _pending_overrides write for both members of
# the offending pair and reruns; _render_time_control picks the override up
# at the very start of its own next call, before that widget is
# instantiated -- the same timing _pending_resets already relies on (see
# _render_time_control's own docstring). See _START_END_PAIRS' own comment
# for why all five pairs need this, not just exercise -- meal/drink fail
# the same silent-no-op way, and a reversed altitude ascent/descent pair is
# worse: it produces an unphysical altitude spike rather than doing
# nothing.
_pair_warnings = []
for _section, _start_name, _end_name in _START_END_PAIRS:
    _start_val = st.session_state.params[_section][_start_name]
    _end_val = st.session_state.params[_section][_end_name]
    if _start_val > _end_val:
        _pair_warnings.append((_section, _start_name, _end_name, _start_val, _end_val))

if _pair_warnings:
    st.sidebar.markdown("---")
    for _section, _start_name, _end_name, _start_val, _end_val in _pair_warnings:
        st.sidebar.warning(
            "**%s.%s** (%.0f min) is after **%s.%s** (%.0f min).  \n"
            "This window will never trigger -- the simulation checks "
            "`%s <= t <= %s`, which is never true when start is after end. "
            "(For the altitude ascent/descent pair specifically, a reversed "
            "pair produces an unphysical altitude spike instead of simply "
            "doing nothing -- see human_body.py's `altitude()`.)"
            % (_section, _start_name, _start_val, _section, _end_name, _end_val,
               _start_name, _end_name)
        )
        if st.sidebar.button(
            "Swap %s.%s ⇄ %s.%s" % (_section, _start_name, _section, _end_name),
            key="swap_%s_%s_%s" % (_section, _start_name, _end_name),
        ):
            st.session_state._pending_overrides[(_section, _start_name)] = _end_val
            st.session_state._pending_overrides[(_section, _end_name)] = _start_val
            st.rerun()
    st.sidebar.markdown("---")

st.sidebar.markdown("---")
_duration_step = HYBRID_DT if SIM_MODE == "hybrid" else 10.0
duration = st.sidebar.number_input(
    "Run duration (minutes)", min_value=10.0, max_value=4320.0,
    value=float(st.session_state.params["simulation"]["duration"]), step=_duration_step,
    help="1440 = 24 hours." + (
        " Kept to multiples of %.0f while hybrid mode is on." % HYBRID_DT
        if SIM_MODE == "hybrid" else ""
    ),
    # Explicit, mode-independent key: `step` above varies with SIM_MODE, and
    # a Streamlit widget with no explicit key derives its identity partly
    # from its own arguments -- letting `step` change would change the
    # widget's identity every time the sidebar mode toggle is flipped,
    # which can drop or reset its state. A fixed key keeps this one widget
    # stable across that toggle.
    key="sidebar_duration",
)
st.session_state.params["simulation"]["duration"] = duration


# ---------------------------------------------------------------------------
# Helpers — coupled body
# ---------------------------------------------------------------------------

def run_body(params_dict, minutes, mode=None):
    """Run the coupled body. mode="hybrid" (default: st.session_state.sim_mode)
    advances kidney/liver/lungs/pancreas with their trained BiLSTM
    surrogates; mode="physics" is the original, unchanged pure-ODE path."""
    mode = mode or SIM_MODE
    p = copy.deepcopy(params_dict)
    if mode == "hybrid":
        p["simulation"]["dt"] = HYBRID_DT
        body = HumanBody(p, surrogate_organs=HYBRID_ORGANS)
        return body.run(minutes=minutes, dt=HYBRID_DT)
    body = HumanBody(p)
    return body.run(minutes=minutes)


def line_panel(ax, df, col, label, color, shade_windows=None):
    ax.plot(df["Time_min"], df[col], color=color, linewidth=1.4)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xlabel("Time (min)", fontsize=8)
    if shade_windows:
        for start, end, c in shade_windows:
            if end > start:
                ax.axvspan(start, end, color=c, alpha=0.12)
    # The caller may put this panel in a sharex=True grid (Run simulation's
    # 3x2 layout does), which hides tick numbers on every row but the
    # bottom one by default -- force them back on so each panel is readable
    # on its own regardless of which row it lands in.
    ax.tick_params(labelsize=8, labelbottom=True)


def disturbance_windows(params_dict):
    pk, pp, pl = params_dict["kidney"], params_dict["pancreas"], params_dict["lungs"]
    windows = [
        (pk["drink_start"], pk["drink_end"], "#2c5282"),
        (pp["meal_start"], pp["meal_end"], "#e08214"),
        (pl["exercise_start"], pl["exercise_end"], "#2f9e44"),
    ]
    return windows


def _mode_caption():
    if SIM_MODE == "hybrid":
        return (
            "Running in **hybrid mode** — kidney, liver, lungs and pancreas "
            "advanced by their trained BiLSTM surrogates once warm-up "
            "(2h) fills; heart is always physics."
        )
    return "Running in **pure physics mode** — every organ solved from its ODE every step."


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Dynamic Multi-Organ State Simulation — Digital Twin")
st.caption(
    "Five coupled organ models (heart, lungs, liver, kidney, pancreas), "
    "%d coupling edges, %d feedback loops. No blood compartment — organs "
    "exchange variables directly, the way the physiology actually connects them."
    % (len(COUPLING_MAP), len(FEEDBACK_LOOPS))
)

(tab_run, tab_ablation, tab_sweep, tab_scenarios, tab_drugs, tab_graph,
 tab_organ) = st.tabs(
    ["Run simulation", "Coupling ablation", "Parameter sweep", "Disease scenarios",
     "Drug interventions", "Coupling graph", "Single organ"]
)

# --- Tab 1: run --------------------------------------------------------

with tab_run:
    st.caption(_mode_caption())
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Run")
        run_clicked = st.button("▶ Run simulation", type="primary", use_container_width=True)
        steady_clicked = st.button(
            "Run steady-state check (4h, disturbances off)", use_container_width=True
        )
        st.caption(
            "Steady-state check runs the current parameters with the meal, "
            "drink, exercise and altitude disturbances switched off, for 4 "
            "hours, and reports whether each variable holds still — the "
            "first test any coupled model has to pass. Always pure "
            "physics, regardless of the sidebar mode toggle: "
            "`HumanBody.steady_state_check` is a fixed built-in method."
        )

    if run_clicked:
        with st.spinner("Integrating %d organ-minutes (%s)..." % (int(duration), SIM_MODE)):
            st.session_state.result_df = run_body(st.session_state.params, duration)

    if steady_clicked:
        quiet = copy.deepcopy(st.session_state.params)
        quiet["pancreas"]["meal_Ra"] = 0.0
        quiet["lungs"]["exercise_factor"] = 1.0
        quiet["kidney"]["drink_rate"] = quiet["kidney"]["baseline_intake"]
        quiet["lungs"]["altitude_target"] = 0.0
        with st.spinner("Running steady-state check..."):
            body = HumanBody(quiet)
            results, quiet_df = body.steady_state_check(minutes=240)
        rows = []
        for var, r in results.items():
            rows.append({
                "variable": var, "start": r["start"], "end": r["end"],
                "drift": r["drift"], "tolerance": r["tolerance"],
                "verdict": "stable" if r["stable"] else "DRIFTS",
            })
        steady_tbl = pd.DataFrame(rows)
        n_stable = int(steady_tbl.verdict.eq("stable").sum())
        with right:
            st.subheader("Steady-state check")
            st.dataframe(steady_tbl, use_container_width=True, hide_index=True)
            st.caption("%d of %d variables hold steady over 4 undisturbed hours." % (n_stable, len(steady_tbl)))

    df = st.session_state.result_df
    if df is not None:
        with right:
            st.subheader("Result — %d minutes, %d steps" % (df.Time_min.max(), len(df)))
            summary_cols = [
                "MAP_mmHg", "Cardiac_Output_L_min", "PaO2_mmHg", "PaCO2_mmHg",
                "Glucose_mg_dL", "Insulin_uU_mL", "Glycogen_Fill_Fraction",
                "Plasma_Volume_L", "BUN_mg_dL",
            ]
            st.dataframe(
                df[summary_cols].describe().T[["min", "mean", "max"]].round(3),
                use_container_width=True,
            )
            if "Kidney_Source" in df.columns:
                source_cols = [c for c in df.columns if c.endswith("_Source")]
                frac_surrogate = {
                    c.replace("_Source", ""): float((df[c] == "surrogate").mean())
                    for c in source_cols
                }
                st.caption(
                    "Fraction of steps advanced by the surrogate (rest is "
                    "physics warm-up): " + ", ".join(
                        "%s %.0f%%" % (o, f * 100) for o, f in frac_surrogate.items()
                    )
                )

            windows = disturbance_windows(st.session_state.params)
            panels = [
                ("MAP_mmHg", "Mean arterial pressure (mmHg)"),
                ("Glucose_mg_dL", "Plasma glucose (mg/dL)"),
                ("PaO2_mmHg", "Arterial O2 (mmHg)"),
                ("Plasma_Volume_L", "Plasma volume (L)"),
                ("Glycogen_Fill_Fraction", "Hepatic glycogen (fraction)"),
                ("BUN_mg_dL", "Blood urea nitrogen (mg/dL)"),
            ]
            fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
            for ax, (col, label) in zip(axes.ravel(), panels):
                line_panel(ax, df, col, label, COLORS.get(col, "#374151"), windows)
            fig.suptitle(
                "blue = drink window, orange = meal window, green = exercise window",
                fontsize=9,
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.download_button(
                "Download this run as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="coupled_body_run.csv",
                mime="text/csv",
            )
    elif not steady_clicked:
        with right:
            st.info("Set parameters in the sidebar, then click Run simulation.")

# --- Tab 2: ablation -----------------------------------------------------

with tab_ablation:
    st.subheader("Coupling ablation")
    st.caption(_mode_caption())
    st.caption(
        "Every coupling has a gain. Setting one to zero disconnects that "
        "edge — the cleanest way to see what it actually contributes: run "
        "the body with it, run without it, compare."
    )
    coupling_names = list(SCHEMA["coupling"].keys())
    col1, col2 = st.columns(2)
    with col1:
        edge_to_cut = st.selectbox(
            "Coupling gain to cut", coupling_names,
            format_func=lambda e: _sweep_param_label(e, _COUPLING_PLAIN_LABELS),
        )
    with col2:
        ablation_minutes = st.number_input(
            "Duration (minutes)", min_value=10.0, max_value=1440.0, value=400.0,
            step=(HYBRID_DT if SIM_MODE == "hybrid" else 10.0),
            key="ablation_minutes",   # stable identity -- see sidebar_duration's comment
        )
    variable_to_watch = st.selectbox(
        "Variable to compare",
        ["MAP_mmHg", "Plasma_Volume_L", "PaO2_mmHg", "Glucose_mg_dL",
         "Cardiac_Output_L_min", "BUN_mg_dL", "GFR_L_min"],
        format_func=_axis_label,
    )

    if SIM_MODE == "hybrid" and edge_to_cut in INERT_UNDER_HYBRID:
        inert_organ, why = INERT_UNDER_HYBRID[edge_to_cut]
        st.warning(
            "**%s is inert under hybrid mode**: cutting it will make little "
            "or no difference to this run, because %s. Switch to Pure "
            "physics in the sidebar to see this edge's real effect." % (edge_to_cut, why)
        )

    if st.button("Run ablation", type="primary"):
        on_params = copy.deepcopy(st.session_state.params)
        off_params = copy.deepcopy(st.session_state.params)
        off_params["coupling"][edge_to_cut] = 0.0

        with st.spinner("Running with and without the coupling..."):
            df_on = run_body(on_params, ablation_minutes)
            df_off = run_body(off_params, ablation_minutes)

        # Two separate panels side by side, one run each, rather than a
        # single overlaid chart -- easier to read the shape of each run on
        # its own, especially when the two lines sit close together. Both
        # panels share one y-axis range (computed from BOTH runs, not
        # auto-scaled independently) so the comparison stays honest: two
        # independently-scaled axes could make curves of very different
        # magnitude look deceptively similar.
        y_lo = min(df_on[variable_to_watch].min(), df_off[variable_to_watch].min())
        y_hi = max(df_on[variable_to_watch].max(), df_off[variable_to_watch].max())
        pad = (y_hi - y_lo) * 0.05 if y_hi > y_lo else max(abs(y_hi), 1.0) * 0.05
        y_lo, y_hi = y_lo - pad, y_hi + pad

        fig, (ax_on, ax_off) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, df, title, color in (
            (ax_on, df_on, "Coupled (%s intact)" % edge_to_cut, "#2c5282"),
            (ax_off, df_off, "%s cut" % edge_to_cut, "#9ca3af"),
        ):
            ax.plot(df.Time_min, df[variable_to_watch], color=color, linewidth=1.6)
            for start, end, c in disturbance_windows(st.session_state.params):
                if end > start:
                    ax.axvspan(start, end, color=c, alpha=0.10)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Time (min)")
            ax.set_ylim(y_lo, y_hi)
            ax.tick_params(labelsize=8)
        ax_on.set_ylabel(_axis_label(variable_to_watch))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        peak_on = df_on[variable_to_watch].max() - df_on[variable_to_watch].iloc[0]
        peak_off = df_off[variable_to_watch].max() - df_off[variable_to_watch].iloc[0]
        st.write(
            "Peak change in **%s**: with the coupling **%+.3f**, with it cut **%+.3f**."
            % (variable_to_watch, peak_on, peak_off)
        )

# --- Tab 3: sweep ----------------------------------------------------------

with tab_sweep:
    st.subheader("Parameter sweep")
    st.caption(_mode_caption())
    st.caption(
        "Sweep one organ's parameter and watch every other organ's end "
        "state respond — the question the coupled model exists to answer."
    )
    section_names = [s for s, _ in SECTION_ORDER if s != "simulation"]
    col1, col2, col3 = st.columns(3)
    with col1:
        sweep_section = st.selectbox("Organ", section_names)
    with col2:
        if sweep_section == "lungs":
            _param_options = [
                p for p in SCHEMA[sweep_section].keys()
                if p not in _LUNGS_SWEEP_EXCLUDE
            ]
        else:
            _param_options = list(SCHEMA[sweep_section].keys())
        _plain_labels = _SWEEP_PLAIN_LABELS.get(sweep_section, {})
        sweep_param = st.selectbox(
            "Parameter", _param_options,
            format_func=lambda p: _sweep_param_label(p, _plain_labels),
        )
    with col3:
        sweep_minutes = st.number_input(
            "Duration (minutes)", min_value=10.0, max_value=1440.0, value=240.0,
            step=(HYBRID_DT if SIM_MODE == "hybrid" else 10.0),
            key="sweep_minutes",
        )

    # A fresh Organ/Parameter selection invalidates any previously computed
    # table immediately -- otherwise the old table (from a different
    # organ/parameter) stays on screen looking exactly like a result for the
    # *new* selection, which is what was actually being reported as "every
    # organ and parameter shows the same table."
    _sweep_key = "%s.%s" % (sweep_section, sweep_param)
    if st.session_state.compare_df_key != _sweep_key:
        st.session_state.compare_df = None

    if SIM_MODE == "hybrid" and sweep_section in HYBRID_ORGANS:
        st.warning(_HYBRID_ORGAN_CAVEAT % (
            sweep_section.capitalize(), sweep_section.capitalize(), sweep_section.capitalize()))

    meta = SCHEMA[sweep_section][sweep_param]
    _is_timing_param = sweep_param in _SWEEP_TIMING_PARAMS
    _default_hi = float(meta["max"])
    if _is_timing_param:
        # This parameter's full schema range is a 0-1440 min clock, almost
        # always wider than the run's own Duration -- default the upper end
        # of the sweep to whichever is smaller, so the default range stays
        # inside the window where the scripted drink/meal event can actually
        # occur, rather than mostly sweeping times the run never reaches.
        _default_hi = min(_default_hi, float(sweep_minutes))
    lo, hi = st.slider(
        "Value range to sweep", min_value=float(meta["min"]), max_value=float(meta["max"]),
        value=(float(meta["min"]), _default_hi),
        # Explicit, selection-scoped key (same pattern as sweep_minutes/
        # ablation_minutes above): guarantees this slider resets to the
        # freshly computed default the moment Organ or Parameter changes,
        # rather than silently keeping a stale range from a previous
        # parameter that happened to share the same auto-generated identity.
        key="sweep_range_%s_%s" % (sweep_section, sweep_param),
    )
    if _is_timing_param and hi > sweep_minutes:
        st.warning(
            "This range goes up to %.0f min, past the %.0f-minute Duration set "
            "above. Any swept value past Duration never actually happens "
            "during the run, so those rows will look flat/unaffected -- "
            "that's the run's own clock running out, not a bug. Raise "
            "Duration or pull the range back in to keep every row "
            "meaningful." % (hi, sweep_minutes)
        )
    n_points = st.slider("Number of values", min_value=2, max_value=8, value=4)

    watch_cols = ["MAP_mmHg", "Cardiac_Output_L_min", "PaO2_mmHg", "PaCO2_mmHg",
                  "Glucose_mg_dL", "Glycogen_Fill_Fraction", "Plasma_Volume_L",
                  "BUN_mg_dL", "GFR_L_min"]

    if st.button("Run sweep", type="primary"):
        values = np.linspace(lo, hi, int(n_points))
        rows = []
        progress = st.progress(0.0)
        for i, v in enumerate(values):
            p = copy.deepcopy(st.session_state.params)
            p[sweep_section][sweep_param] = float(v)
            d = run_body(p, sweep_minutes)
            row = {"%s.%s" % (sweep_section, sweep_param): float(v)}
            row.update({c: float(d[c].iloc[-1]) for c in watch_cols})
            rows.append(row)
            progress.progress((i + 1) / len(values))
        sweep_df = pd.DataFrame(rows)
        st.session_state.compare_df = sweep_df
        st.session_state.compare_df_key = _sweep_key

    if st.session_state.compare_df is not None:
        sweep_df = st.session_state.compare_df
        st.dataframe(sweep_df, use_container_width=True, hide_index=True)
        x_col = sweep_df.columns[0]
        x_label = "%s: %s" % (sweep_section.capitalize(), _sweep_param_label(sweep_param, _plain_labels))
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        for ax, col in zip(axes.ravel(), watch_cols):
            ax.plot(sweep_df[x_col], sweep_df[col], "o-", color=COLORS.get(col, "#374151"))
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel(_axis_label(col), fontsize=8)
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.download_button(
            "Download sweep as CSV",
            data=sweep_df.to_csv(index=False).encode("utf-8"),
            file_name="sweep_%s_%s.csv" % (sweep_section, sweep_param),
            mime="text/csv",
        )
    else:
        st.caption("No sweep run yet for this organ/parameter — click **Run sweep** above.")

# --- Tab 4: disease scenarios ------------------------------------------------

# Three-way verdict scheme for organ_impact_summary() above -- deliberately
# different from drug interventions' four-way Positive/Negative/Balanced/No
# effect scheme (see disease_scenarios.py's own docstring for why: a
# disease preset has no "desired" vs. "side effect" framing to judge
# against, only "did the coupling carry this to another organ").
_DISEASE_VERDICT_CALLOUT = {
    "Directly affected":         st.error,
    "Downstream effect":         st.warning,
    "Not meaningfully affected": st.info,
}
_DISEASE_VERDICT_PHRASE = {
    "Directly affected":         "a directly affected organ",
    "Downstream effect":         "a downstream (coupling) effect",
    "Not meaningfully affected": "no meaningful effect",
}

with tab_scenarios:
    st.subheader("Disease scenarios")
    st.caption(
        "Always pure physics, regardless of the sidebar mode toggle. Each "
        "preset works by changing an organ's ODE parameters (e.g. CKD "
        "raises kidney RVR); a surrogate's prediction depends only on its "
        "trained weights and its own recent trajectory, never on those "
        "parameters, so hybrid mode would silently leave a "
        "surrogate-driven organ's state unmoved by the very change each "
        "scenario exists to demonstrate. See app.py's module docstring for "
        "the full explanation."
    )
    st.caption(
        "Three illustrative disease presets — T2D, CKD, CHF — built on the "
        "same organ ODEs and coupling graph as everywhere else in this app, "
        "just started from a different physiological baseline."
    )

    scenario_names = list(SCENARIOS.keys())
    col1, col2 = st.columns(2)
    with col1:
        scenario_choice = st.selectbox(
            "Scenario", scenario_names,
            format_func=lambda n: "%s — %s" % (n, SCENARIOS[n]["label"]),
        )
    with col2:
        scenario_minutes = st.number_input(
            "Duration (minutes)", min_value=60.0, max_value=4320.0, value=1440.0,
            step=60.0, key="scenario_minutes",
        )

    use_sidebar_baseline = st.checkbox(
        "Apply on top of my current sidebar configuration",
        value=True,
        help=(
            "Checked: the scenario's overrides are layered on top of "
            "whatever you've set in the sidebar, so both the baseline and "
            "scenario runs use your parameters. Unchecked: both runs use "
            "the model's original defaults, ignoring the sidebar."
        ),
    )

    with st.expander("What this scenario changes", expanded=False):
        meta = SCENARIOS[scenario_choice]
        for organ, changes in meta["overrides"].items():
            for param, value in changes.items():
                st.write("- **%s.%s** → %s" % (organ, param, value))
        st.write("Expected effects:")
        for col, direction, rationale in meta["expect"]:
            st.write("- **%s** should go **%s** — %s" % (col, direction, rationale))

    if st.button("▶ Run scenario", type="primary"):
        base_params = copy.deepcopy(st.session_state.params) if use_sidebar_baseline else None
        with st.spinner("Running baseline and %s..." % scenario_choice):
            baseline_df, scenario_df = run_scenario(
                scenario_choice, minutes=scenario_minutes, base_params=base_params,
            )
            effects_df = evaluate_effects(baseline_df, scenario_df, meta["expect"])
            organ_summary = disease_organ_impact_summary(scenario_choice, baseline_df, scenario_df)
        st.session_state.scenario_result = {
            "name": scenario_choice,
            "baseline_df": baseline_df,
            "scenario_df": scenario_df,
            "effects_df": effects_df,
            "organ_summary": organ_summary,
        }

    result = st.session_state.scenario_result
    if result is not None:
        st.markdown("---")
        st.subheader("%s vs. baseline" % result["name"])

        n_holds = int(result["effects_df"]["holds"].sum())
        n_total = len(result["effects_df"])
        st.dataframe(result["effects_df"], use_container_width=True, hide_index=True)
        if n_holds == n_total:
            st.success("%d of %d expected effects held." % (n_holds, n_total))
        else:
            st.warning("%d of %d expected effects held." % (n_holds, n_total))

        watch_cols = [row["variable"] for row in result["effects_df"].to_dict("records")]
        # De-duplicate while preserving order, in case a scenario's expect
        # list ever repeats a column.
        seen = set()
        watch_cols = [c for c in watch_cols if not (c in seen or seen.add(c))]

        n_cols = 2
        n_rows = int(np.ceil(len(watch_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, watch_cols):
            ax.plot(
                result["baseline_df"]["Time_min"], result["baseline_df"][col],
                color="#9ca3af", linewidth=1.4, label="baseline",
            )
            ax.plot(
                result["scenario_df"]["Time_min"], result["scenario_df"][col],
                color=COLORS.get(col, "#c0392b"), linewidth=1.6, label=result["name"],
            )
            ax.set_ylabel(_axis_label(col), fontsize=9)
            ax.set_xlabel("Time (min)", fontsize=8)
            # sharex=True hides tick numbers on every row but the bottom
            # one by default -- force them back on for every panel so each
            # chart is readable on its own, not just the last row of the grid.
            ax.tick_params(labelsize=8, labelbottom=True)
            ax.legend(fontsize=7)
        for ax in axes[len(watch_cols):]:
            ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Cross-organ effects")
        st.caption(
            "Always pure physics, regardless of the sidebar mode toggle -- "
            "same reason as the caption at the top of this tab: this "
            "scenario works by changing an organ's ODE parameters directly, "
            "and a BiLSTM surrogate's prediction never reads those "
            "parameters, so hybrid mode would leave a surrogate-driven "
            "organ's state unmoved by the very change being demonstrated."
        )
        st.caption(
            "Every one of the five organs, not just the columns in the "
            "table above -- so a coupling effect on an organ this "
            "scenario's own expect list doesn't mention is still visible."
        )
        for o, info in result["organ_summary"].items():
            verdict = info["verdict"]
            callout = _DISEASE_VERDICT_CALLOUT[verdict]
            phrase = _DISEASE_VERDICT_PHRASE[verdict]
            pct = info["max_abs_pct_change"]
            if pct > 150.0:
                change_note = "largest change: very large relative to a near-zero baseline value"
            else:
                change_note = "largest change: %.1f%%" % pct
            col_bits = ", ".join(
                "%s %+.1f%%" % (_axis_label(c), v) for c, v in info["column_pct_changes"].items()
            )
            message = "**%s** on the **%s** — %s (%s)." % (
                phrase.capitalize(), ORGAN_LABELS[o], change_note, col_bits,
            )
            callout(message)

        # Full multi-panel graph across every organ's own signature columns
        # (not just the columns in this scenario's own expect list) -- the
        # same "show every organ, not just the target" idea as the callouts
        # above, in chart form.
        all_organ_cols = list(dict.fromkeys(
            c for cols in DISEASE_ORGAN_SIGNATURE_COLUMNS.values() for c in cols
        ))
        n_cols = 3
        n_rows = int(np.ceil(len(all_organ_cols) / n_cols))
        # Which organ each column belongs to, so every panel can carry a
        # small "(Organ)" title -- the grid otherwise mixes all 5 organs'
        # columns with no visual grouping.
        _col_to_organ = {
            c: organ for organ, cols in DISEASE_ORGAN_SIGNATURE_COLUMNS.items() for c in cols
        }
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.6 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, all_organ_cols):
            ax.plot(
                result["baseline_df"]["Time_min"], result["baseline_df"][col],
                color="#9ca3af", linewidth=1.3, label="baseline",
            )
            ax.plot(
                result["scenario_df"]["Time_min"], result["scenario_df"][col],
                color=COLORS.get(col, "#c0392b"), linewidth=1.5, label=result["name"],
            )
            organ_name = ORGAN_LABELS.get(_col_to_organ.get(col), "")
            ax.set_title(organ_name, fontsize=8, color="#6b7280")
            ax.set_ylabel(_axis_label(col), fontsize=8)
            ax.set_xlabel("Time (minutes)", fontsize=7)
            # sharex=True hides tick numbers on every row but the bottom
            # one by default -- force them back on for every panel (this is
            # the fix for the missing numeric ticks on the upper rows).
            ax.tick_params(labelsize=7, labelbottom=True)
            ax.legend(fontsize=6)
        for ax in axes[len(all_organ_cols):]:
            ax.axis("off")
        fig.suptitle("Every organ's signature variables, baseline vs. %s" % result["name"], fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download baseline run as CSV",
                data=result["baseline_df"].to_csv(index=False).encode("utf-8"),
                file_name="scenario_%s_baseline.csv" % result["name"],
                mime="text/csv",
            )
        with dl2:
            st.download_button(
                "Download %s run as CSV" % result["name"],
                data=result["scenario_df"].to_csv(index=False).encode("utf-8"),
                file_name="scenario_%s.csv" % result["name"],
                mime="text/csv",
            )
    else:
        st.info("Pick a scenario and click Run scenario.")

# --- Tab 5: drug interventions ------------------------------------------

_VERDICT_CALLOUT = {
    "Positive":   st.success,
    "Negative":   st.error,
    "Balanced":   st.warning,
    "No effect":  st.info,
}
_VERDICT_PHRASE = {
    "Positive":  "a **positive** impact",
    "Negative":  "a **negative** impact",
    "Balanced":  "a **balanced / mixed** impact",
    "No effect": "**no meaningful effect**",
}

_EVAL_COLUMN_LABELS = {
    "variable": "Variable",
    "organ": "Organ",
    "desired": "Effect type",
    "expected": "Expected direction",
    "eval_time_min": "Eval time (min)",
    "baseline_value": "Baseline",
    "treated_value": "Treated",
    "delta": "Change (Δ)",
    "holds": "Held?",
    "rationale": "Rationale",
}


def _render_html_table_wrapped(df, css_class, last_col_min_px=260, last_col_max_px=460):
    """st.dataframe truncates long text with an ellipsis and gives no way to
    read a full sentence-length cell without clicking into it one at a time.
    Render a plain HTML table instead, with every cell (and especially the
    last column, usually the longest free-text one) explicitly allowed to
    wrap onto multiple lines. Shared by the evaluation table and the static
    reference-medications table below.

    NOTE: every line of the CSS below starts in column 0 -- st.markdown's
    renderer treats 4+ leading spaces of indentation as a code block, which
    would print this HTML/CSS as literal text instead of rendering it. And
    the concatenation below is deliberate, not %-formatting -- the CSS
    contains literal '%' characters (e.g. "width: 100%") that a %-format
    string would misparse as format specifiers.
    """
    html_table = df.to_html(index=False, escape=True, border=0)
    style_block = (
        "<style>"
        ".%s { overflow-x: auto; margin-bottom: 0.5rem; }" % css_class +
        ".%s table { width: 100%%; border-collapse: collapse; font-size: 0.85rem; }" % css_class +
        ".%s th, .%s td {" % (css_class, css_class) +
        "padding: 6px 10px; border: 1px solid rgba(128, 128, 128, 0.35);"
        "text-align: left; vertical-align: top; white-space: normal; word-wrap: break-word;"
        "}" +
        ".%s th { font-weight: 600; }" % css_class +
        ".%s td:last-child, .%s th:last-child {" % (css_class, css_class) +
        "min-width: %dpx; max-width: %dpx;" % (last_col_min_px, last_col_max_px) +
        "}" +
        "</style>"
    )
    st.markdown(style_block + ('<div class="%s">' % css_class) + html_table + "</div>", unsafe_allow_html=True)


def _render_wrapped_evaluation_table(evaluation_df):
    df = evaluation_df.copy()
    df["desired"] = df["desired"].map({True: "Intended effect", False: "Side effect (watched for)"})
    df["holds"] = df["holds"].map({True: "✅ Yes", False: "❌ No"})
    df["expected"] = df["expected"].str.capitalize()
    df["organ"] = df["organ"].map(lambda o: ORGAN_LABELS.get(o, o))
    df = df.rename(columns=_EVAL_COLUMN_LABELS)
    _render_html_table_wrapped(df, "wrapped-eval-table")


with tab_drugs:
    st.subheader("Drug interventions")
    st.caption(
        "Always pure physics, regardless of the sidebar mode toggle, for "
        "the same reason as Disease scenarios above: a drug here acts by "
        "nudging an organ's ODE parameter over time (see "
        "`HumanBodyWithInterventions.step` in drug_interventions.py), and "
        "a surrogate-driven organ's prediction never reads that parameter."
    )
    st.caption(
        "Fourteen illustrative drugs, each acting on one or more organ "
        "parameters shown against it below, dosed as a single "
        "administration with a real pharmacokinetic rise-peak-decay shape "
        "(not a permanent change). Organised by the organ each drug "
        "primarily targets -- pick an organ tab to see what's modelled "
        "for it."
    )

    subtab_single, subtab_multi = st.tabs(["Single drug", "Multi-drug testing"])

    with subtab_single:
        organ_tabs = st.tabs([ORGAN_LABELS[o] for o in ORGAN_SIGNATURE_COLUMNS])

        for organ_tab, organ in zip(organ_tabs, ORGAN_SIGNATURE_COLUMNS.keys()):
            with organ_tab:
                drug_names = ORGAN_DRUGS[organ]
                if drug_names:
                    if len(drug_names) == 1:
                        drug_name = drug_names[0]
                        st.caption("1 drug simulated for the %s." % ORGAN_LABELS[organ].lower())
                    else:
                        drug_name = st.selectbox(
                            "Choose a drug to simulate for the %s" % ORGAN_LABELS[organ].lower(),
                            drug_names,
                            format_func=lambda n: DRUGS[n]["label"],
                            key="drugselect_%s" % organ,
                        )
                    drug = DRUGS[drug_name]
                    st.markdown("### %s" % drug["label"])
                    st.caption("%s -- primarily acts on the %s" % (drug["drug_class"], ORGAN_LABELS[drug["primary_organ"]].lower()))
                    st.markdown("**Commonly used for:** %s" % drug["common_uses"])

                    lo, hi = drug["dose_range_mg"]
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        dose_mg = st.slider(
                            "Dose (mg)", min_value=float(lo), max_value=float(hi),
                            value=float(drug["standard_dose_mg"]),
                            step=float(round((hi - lo) / 30.0, 2)),
                            key="dose_%s" % drug_name,
                            help="Effect scales linearly from the documented standard dose "
                                 "(%.1f mg), clamped so no dose can push the target parameter "
                                 "past a safe physiological bound." % drug["standard_dose_mg"],
                        )
                    with col2:
                        baseline_options = ["Healthy defaults", "My sidebar configuration"]
                        suggested = drug["suggested_scenario"]
                        if suggested:
                            baseline_options.append("%s baseline (suggested)" % suggested)
                        default_idx = len(baseline_options) - 1 if suggested else 0
                        baseline_choice = st.selectbox(
                            "Test against", baseline_options, index=default_idx,
                            key="baseline_%s" % drug_name,
                        )
                    with col3:
                        drug_minutes = st.number_input(
                            "Duration (min)", min_value=120.0, max_value=4320.0, value=1440.0,
                            step=60.0, key="minutes_%s" % drug_name,
                        )

                    if baseline_choice == "My sidebar configuration":
                        base_params, scenario = copy.deepcopy(st.session_state.params), None
                    elif suggested and baseline_choice.startswith(suggested):
                        base_params, scenario = None, suggested
                    else:
                        base_params, scenario = None, None

                    if st.button("▶ Administer %s" % drug["generic_name"], type="primary", key="run_%s" % drug_name):
                        with st.spinner("Running baseline and %s at %.0f mg..." % (drug["generic_name"], dose_mg)):
                            baseline_df, treated_df = run_intervention(
                                drug_name, minutes=drug_minutes, base_params=base_params,
                                scenario=scenario, dose_mg=dose_mg,
                            )
                            eval_time = drug["t_admin"] + peak_time(drug["ka"], drug["ke"])
                            eval_time = min(eval_time, drug_minutes)
                            evaluation_df = evaluate_intervention(baseline_df, treated_df, drug["expect"], eval_time)
                            organ_summary = organ_impact_summary(drug, baseline_df, treated_df, evaluation_df, eval_time)
                        st.session_state.drug_result[drug_name] = {
                            "dose_mg": dose_mg, "baseline_df": baseline_df, "treated_df": treated_df,
                            "evaluation_df": evaluation_df, "eval_time": eval_time, "organ_summary": organ_summary,
                        }

                    result = st.session_state.drug_result.get(drug_name)
                    if result is None:
                        st.info("Set a dose and baseline, then click Administer.")
                    else:
                        st.markdown("---")
                        st.write(
                            "**For medical professionals** -- expected effects at peak concentration "
                            "(t = %.0f min, dose %.0f mg):" % (result["eval_time"], result["dose_mg"])
                        )
                        _render_wrapped_evaluation_table(result["evaluation_df"])

                        # comparison plot: every column named in this drug's expect list
                        watch_cols = list(dict.fromkeys(e["column"] for e in drug["expect"]))
                        n_cols = 2
                        n_rows = int(np.ceil(len(watch_cols) / n_cols))
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows), sharex=True)
                        axes = np.atleast_1d(axes).ravel()
                        for ax, col in zip(axes, watch_cols):
                            ax.plot(result["baseline_df"]["Time_min"], result["baseline_df"][col],
                                    color="#9ca3af", linewidth=1.4, label="no drug")
                            ax.plot(result["treated_df"]["Time_min"], result["treated_df"][col],
                                    color=COLORS.get(col, "#c0392b"), linewidth=1.6, label=drug["generic_name"])
                            ax.axvline(result["eval_time"], color="#374151", linestyle=":", linewidth=1.0)
                            ax.set_ylabel(_axis_label(col), fontsize=9)
                            ax.set_xlabel("Time (min)", fontsize=8)
                            # sharex=True hides tick numbers on every row but the
                            # bottom one by default -- force them back on for
                            # every panel.
                            ax.tick_params(labelsize=8, labelbottom=True)
                            ax.legend(fontsize=7)
                        for ax in axes[len(watch_cols):]:
                            ax.axis("off")
                        fig.suptitle("dotted line = time of peak drug concentration", fontsize=9)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                        dl1, dl2 = st.columns(2)
                        with dl1:
                            st.download_button(
                                "Download baseline (no drug) run as CSV",
                                data=result["baseline_df"].to_csv(index=False).encode("utf-8"),
                                file_name="%s_baseline.csv" % drug_name, mime="text/csv",
                                key="dl_base_%s" % drug_name,
                            )
                        with dl2:
                            st.download_button(
                                "Download %s run as CSV" % drug["generic_name"],
                                data=result["treated_df"].to_csv(index=False).encode("utf-8"),
                                file_name="%s_treated.csv" % drug_name, mime="text/csv",
                                key="dl_treat_%s" % drug_name,
                            )

                        st.markdown("#### AI's point of view")
                        st.caption(
                            "A plain-language read of the same table above -- what this drug "
                            "did to each organ, including ones it doesn't directly target, "
                            "so the effect on related and co-related organs is clear at a glance. "
                            "Where movement is meaningful, a short note explains what kind of "
                            "effect or problem it could reflect in the human body, and, where a "
                            "real, well-documented complication exists, a second note names what "
                            "it could lead to if severe or unmanaged (e.g. kidney failure, a heart "
                            "attack) -- not a prediction for this run, just the real-world stakes "
                            "behind the mechanism above it."
                        )
                        for o, info in result["organ_summary"].items():
                            verdict = info["verdict"]
                            callout = _VERDICT_CALLOUT[verdict]
                            phrase = _VERDICT_PHRASE[verdict]
                            pct = info["max_abs_pct_change"]
                            # A signature column with a near-zero baseline (e.g.
                            # resting Urine_Output_L_min) makes an ordinary percent
                            # change look enormous even for a modest absolute
                            # change -- flag that rather than print a misleading
                            # number.
                            if pct > 150.0:
                                change_note = "largest change: very large relative to a near-zero baseline value"
                            else:
                                change_note = "largest change: %.1f%%" % pct
                            message = "**%s** has %s on the **%s** (%s)." % (
                                drug["label"], phrase, ORGAN_LABELS[o], change_note,
                            )
                            # Plain-language "what this might mean" -- only shown
                            # when there's a meaningful effect to explain and the
                            # drug has a written note for this organ. "No effect"
                            # organs don't get one: there's nothing to explain.
                            note = drug["organ_notes"].get(o)
                            if verdict != "No effect" and note:
                                message += "\n\n*What this could mean:* %s" % note
                            # A second, separate line naming the concrete
                            # real-world complication this kind of effect can
                            # progress to if severe or unmanaged (e.g. acute
                            # kidney injury, a heart attack, liver failure) --
                            # deliberately kept apart from "what this could
                            # mean" above, which explains the mechanism/risk
                            # factor rather than the worst-case outcome itself.
                            # Only shown for drugs where that outcome is real
                            # and well-documented, not invented for effect.
                            risk = drug.get("organ_risks", {}).get(o)
                            if verdict != "No effect" and risk:
                                message += "\n\n*What it could lead to:* %s" % risk
                            callout(message)
                else:
                    st.info(
                        "No drug intervention is simulated for the %s in this "
                        "version." % ORGAN_LABELS[organ].lower()
                    )

    with subtab_multi:
        st.markdown(
            "Pick as many drugs as apply, across as many organs as apply -- "
            "e.g. a heart drug **and** a kidney drug together, the way a "
            "real patient is often on more than one prescription at once. "
            "Every organ defaults to **N/A**; leave it there if this "
            "patient isn't taking anything for that organ."
        )
        st.caption(
            "Always pure physics, regardless of the sidebar mode toggle -- "
            "same reason as the Single drug tab and Disease scenarios: "
            "these drugs act by nudging ODE parameters directly, which a "
            "BiLSTM surrogate's prediction never reads."
        )

        st.session_state.setdefault(
            "md_slot_counts", {organ: 1 for organ in ORGAN_DRUGS},
        )

        _NA = "N/A"
        selections_by_organ = {}   # organ -> [(drug_name, dose_mg), ...]
        for organ in ORGAN_DRUGS:
            drug_names_for_organ = ORGAN_DRUGS[organ]
            if not drug_names_for_organ:
                continue
            st.markdown("**%s**" % ORGAN_LABELS[organ])
            n_slots = st.session_state.md_slot_counts[organ]
            organ_picks = []
            for i in range(n_slots):
                slot_cols = st.columns([3, 2])
                with slot_cols[0]:
                    choice = st.selectbox(
                        "Drug %d for the %s" % (i + 1, ORGAN_LABELS[organ].lower()),
                        [_NA] + drug_names_for_organ,
                        format_func=lambda n: "N/A (none)" if n == _NA else DRUGS[n]["label"],
                        key="md_drug_%s_%d" % (organ, i),
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                if choice != _NA:
                    drug = DRUGS[choice]
                    lo, hi = drug["dose_range_mg"]
                    with slot_cols[1]:
                        dose = st.slider(
                            "Dose (mg)", min_value=float(lo), max_value=float(hi),
                            value=float(drug["standard_dose_mg"]),
                            step=float(round((hi - lo) / 30.0, 2)),
                            key="md_dose_%s_%d" % (organ, i),
                            label_visibility="collapsed" if i > 0 else "visible",
                        )
                    organ_picks.append((choice, dose))
            selections_by_organ[organ] = organ_picks

            add_col, remove_col = st.columns([1, 1])
            with add_col:
                if st.button(
                    "+ Add another %s drug" % ORGAN_LABELS[organ].lower(),
                    key="md_add_%s" % organ,
                ):
                    st.session_state.md_slot_counts[organ] += 1
                    st.rerun()
            with remove_col:
                if n_slots > 1 and st.button(
                    "- Remove last %s slot" % ORGAN_LABELS[organ].lower(),
                    key="md_remove_%s" % organ,
                ):
                    st.session_state.md_slot_counts[organ] -= 1
                    # Drop the removed slot's own widget state so it doesn't
                    # reappear pre-filled if a slot is added back later.
                    st.session_state.pop("md_drug_%s_%d" % (organ, n_slots - 1), None)
                    st.session_state.pop("md_dose_%s_%d" % (organ, n_slots - 1), None)
                    st.rerun()
            st.markdown("---")

        all_drug_doses = [
            pick for picks in selections_by_organ.values() for pick in picks
        ]

        if all_drug_doses:
            st.write(
                "**Simulating together:** " + ", ".join(
                    "%s (%s, %.0f mg)" % (DRUGS[n]["label"], ORGAN_LABELS[DRUGS[n]["primary_organ"]].lower(), d)
                    for n, d in all_drug_doses
                )
            )
            # This model applies each selected drug's parameter nudge
            # completely independently (HumanBodyWithInterventions.step,
            # see its own docstring) -- there is no drug-drug interaction
            # term. If two selected drugs happen to target the exact same
            # organ parameter, the engine does not stack/add their effects;
            # whichever is processed later in the list simply overwrites
            # the earlier one's value for that parameter each step. Flagged
            # here rather than silently shown as a combined effect.
            _target_map = {}
            for n, _dose in all_drug_doses:
                for t in DRUGS[n]["targets"]:
                    _target_map.setdefault((t["organ"], t["attr"]), []).append(DRUGS[n]["label"])
            _colliding = {k: v for k, v in _target_map.items() if len(v) > 1}
            if _colliding:
                st.warning(
                    "These selected drugs target the exact same parameter: " +
                    "; ".join("%s.%s <- %s" % (organ, attr, " & ".join(labels))
                              for (organ, attr), labels in _colliding.items()) +
                    ". This model does not stack simultaneous doses on the same "
                    "parameter -- only the last one applied each step wins, so "
                    "the result below is not a combined/additive effect for "
                    "that parameter specifically."
                )
        else:
            st.info("Select at least one drug above to run a multi-drug test.")

        col1, col2 = st.columns(2)
        with col1:
            md_baseline_choice = st.selectbox(
                "Test against", ["Healthy defaults", "My sidebar configuration"],
                key="md_baseline_choice",
            )
        with col2:
            md_minutes = st.number_input(
                "Duration (min)", min_value=120.0, max_value=4320.0, value=1440.0,
                step=60.0, key="md_minutes",
            )

        if st.button(
            "▶ Run multi-drug test", type="primary", key="md_run",
            disabled=not all_drug_doses,
        ):
            md_base_params = (
                copy.deepcopy(st.session_state.params)
                if md_baseline_choice == "My sidebar configuration" else None
            )
            with st.spinner("Running baseline and %d drug(s) together..." % len(all_drug_doses)):
                md_baseline_df, md_treated_df = run_multi_intervention(
                    all_drug_doses, minutes=md_minutes, base_params=md_base_params,
                )
                md_organ_summary = organ_impact_summary_multidrug(
                    [n for n, _ in all_drug_doses], md_baseline_df, md_treated_df,
                )
            st.session_state.multi_drug_result = {
                "drug_doses": all_drug_doses,
                "baseline_df": md_baseline_df,
                "treated_df": md_treated_df,
                "organ_summary": md_organ_summary,
            }

        md_result = st.session_state.get("multi_drug_result")
        if md_result is not None:
            st.markdown("---")
            st.markdown("#### Cross-organ effects")
            st.caption(
                "Every one of the five organs, using the peak deviation "
                "reached anywhere over the whole run -- not one fixed "
                "instant -- since different drugs in this mix peak at "
                "different times, so there's no single moment that fairly "
                "represents the combination the way one drug's own peak "
                "time represents it alone."
            )
            for o, info in md_result["organ_summary"].items():
                verdict = info["verdict"]
                callout = _DISEASE_VERDICT_CALLOUT[verdict]
                phrase = _DISEASE_VERDICT_PHRASE[verdict]
                pct = info["max_abs_pct_change"]
                if pct > 150.0:
                    change_note = "largest change: very large relative to a near-zero baseline value"
                else:
                    change_note = "largest change: %.1f%% (at t=%.0f min)" % (pct, info["peak_time_min"])
                col_bits = ", ".join(
                    "%s %+.1f%%" % (_axis_label(c), v) for c, v in info["column_pct_changes"].items()
                )
                message = "**%s** on the **%s** — %s (%s)." % (
                    phrase.capitalize(), ORGAN_LABELS[o], change_note, col_bits,
                )
                callout(message)

            all_organ_cols = list(dict.fromkeys(
                c for cols in ORGAN_SIGNATURE_COLUMNS.values() for c in cols
            ))
            _col_to_organ_md = {
                c: organ for organ, cols in ORGAN_SIGNATURE_COLUMNS.items() for c in cols
            }
            n_cols = 3
            n_rows = int(np.ceil(len(all_organ_cols) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.6 * n_rows), sharex=True)
            axes = np.atleast_1d(axes).ravel()
            drug_label_str = " + ".join(DRUGS[n]["generic_name"] for n, _ in md_result["drug_doses"])
            for ax, col in zip(axes, all_organ_cols):
                ax.plot(
                    md_result["baseline_df"]["Time_min"], md_result["baseline_df"][col],
                    color="#9ca3af", linewidth=1.3, label="no drugs",
                )
                ax.plot(
                    md_result["treated_df"]["Time_min"], md_result["treated_df"][col],
                    color=COLORS.get(col, "#c0392b"), linewidth=1.5, label=drug_label_str,
                )
                organ_name = ORGAN_LABELS.get(_col_to_organ_md.get(col), "")
                ax.set_title(organ_name, fontsize=8, color="#6b7280")
                ax.set_ylabel(_axis_label(col), fontsize=8)
                ax.set_xlabel("Time (minutes)", fontsize=7)
                ax.tick_params(labelsize=7, labelbottom=True)
                ax.legend(fontsize=6)
            for ax in axes[len(all_organ_cols):]:
                ax.axis("off")
            fig.suptitle("Every organ's signature variables, no drugs vs. %s" % drug_label_str, fontsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download baseline (no drugs) run as CSV",
                    data=md_result["baseline_df"].to_csv(index=False).encode("utf-8"),
                    file_name="multidrug_baseline.csv", mime="text/csv", key="md_dl_base",
                )
            with dl2:
                st.download_button(
                    "Download multi-drug run as CSV",
                    data=md_result["treated_df"].to_csv(index=False).encode("utf-8"),
                    file_name="multidrug_treated.csv", mime="text/csv", key="md_dl_treat",
                )

# --- Tab 6: coupling graph --------------------------------------------------

with tab_graph:
    st.subheader("Coupling graph — %d edges, %d feedback loops" % (len(COUPLING_MAP), len(FEEDBACK_LOOPS)))
    edge_df = pd.DataFrame(COUPLING_MAP, columns=["from", "to", "mechanism"])
    # COUPLING_MAP's 9 rows are in a fixed order (see human_body.py), each
    # driven by exactly one named gain in PARAMETERS["coupling"] -- listed
    # here in that same order (three lungs->heart rows share a (from, to)
    # pair but each has its own gain, so this has to be positional, not a
    # (from, to) lookup). Cross-checked against SCHEMA["coupling"]'s own
    # key set below, so a coupling.py edit that adds/removes/reorders an
    # edge fails loudly here instead of silently mislabelling a row.
    _EDGE_GAIN_NAMES = [
        "map_to_kidney_gain", "preload_gain", "glucose_to_liver_gain",
        "hepatic_flux_to_glucose_gain", "perfusion_dl_exponent",
        "hypoxia_sv_gain", "sympathetic_hr_gain", "co2_vasodilation_gain",
        "hepatic_vo2_gain",
    ]
    if len(_EDGE_GAIN_NAMES) == len(COUPLING_MAP) and set(_EDGE_GAIN_NAMES) <= set(SCHEMA["coupling"]):
        edge_df["gain"] = _EDGE_GAIN_NAMES
        edge_df["inert under hybrid (surrogate mode)"] = [
            ("yes — " + INERT_UNDER_HYBRID[g][1]) if g in INERT_UNDER_HYBRID else "no"
            for g in _EDGE_GAIN_NAMES
        ]
    else:
        # COUPLING_MAP no longer has the shape this table assumed -- show
        # the edges plainly rather than a table that might mislabel one.
        st.warning(
            "COUPLING_MAP has changed shape since this tab's edge/gain "
            "list was written -- showing edges without the gain/hybrid "
            "columns until app.py's _EDGE_GAIN_NAMES is updated to match."
        )
    st.dataframe(edge_df, use_container_width=True, hide_index=True)
    st.caption(
        "\"Inert under hybrid mode\" means: while the destination organ is "
        "running as a BiLSTM surrogate (Run simulation / Coupling ablation "
        "/ Parameter sweep, hybrid mode on), changing this edge's gain has "
        "no effect on that organ's predicted state — see this tab's own "
        "warnings in the Coupling ablation tab, and app.py's module "
        "docstring, for the code-level reason."
    )
    st.subheader("Feedback loops")
    for name, path in FEEDBACK_LOOPS.items():
        st.write("**%s**: %s" % (name, " → ".join(path)))

# --- Tab 7: single organ -----------------------------------------------------

_ORGAN_TIME_UNIT = {
    "heart": ("Time_sec", "s"), "lungs": ("Time_min", "min"), "liver": ("Time_min", "min"),
    "kidney": ("Time_min", "min"), "pancreas": ("Time_min", "min"),
}
_ORGAN_PANELS = {
    "heart": [("Arterial_Pressure_mmHg", "Arterial pressure (mmHg)"),
              ("Blood_Flow_mL_s", "Aortic inflow (mL/s)")],
    "lungs": [("PaO2_mmHg", "Arterial O2 (mmHg)"), ("PaCO2_mmHg", "Arterial CO2 (mmHg)"),
              ("SaO2_percent", "O2 saturation (%)"), ("Altitude_m", "Altitude (m)")],
    "liver": [("Glycogen_Fill_Fraction", "Glycogen store (fraction)"),
              ("HGP_mg_min", "Hepatic glucose production (mg/min)"),
              ("HGU_mg_min", "Hepatic glucose uptake (mg/min)"),
              ("Hepatic_VO2_mL_min", "Hepatic O2 use (mL/min)")],
    "kidney": [("Plasma_Volume_L", "Plasma volume (L)"), ("BUN_mg_dL", "Blood urea nitrogen (mg/dL)"),
               ("GFR_L_min", "GFR (L/min)"), ("Urine_Output_L_min", "Urine output (L/min)")],
    "pancreas": [("Glucose_mg_dL", "Plasma glucose (mg/dL)"), ("Insulin_uU_mL", "Plasma insulin (uU/mL)"),
                 ("Insulin_Action_1_min", "Insulin action (1/min)"), ("Meal_Ra_mg_min", "Gut glucose input (mg/min)")],
}
_ORGAN_KNOWN_LIMITATIONS = {
    "pancreas": (
        "Known limitation, documented in ode_models/Phy_pancreases_ode.py: "
        "the secretion term's time factor is Bergman's original IVGTT "
        "formulation (t since the glucose bolus), applied here to absolute "
        "simulation time instead. Over a 24h run this over-drives insulin "
        "secretion and can push glucose into hypoglycaemic range in some "
        "parameter combinations -- a modelling issue tracked for the M2 "
        "calibration step, not a bug in this tab."
    ),
}

with tab_organ:
    st.subheader("Single organ")
    st.caption(
        "One organ, alone, always pure ODE physics -- no BiLSTM surrogate "
        "anywhere in this tab. Runs from `organ_models/<organ>_artifact.pkl` "
        "(export_organ_models.py), each organ's own PARAMETERS slice, "
        "unrelated to the coupled body's model_artifact.pkl above."
    )

    _ORGAN_ORDER = ["heart", "lungs", "liver", "kidney", "pancreas"]
    # One sub-tab per organ, every one of them rendered on every run (the
    # same pattern the Drug interventions tab above already uses for its
    # per-organ sub-tabs) -- st.tabs only ever *hides* the inactive tabs'
    # content in the browser, it doesn't skip creating their widgets, so
    # every organ's sliders keep a stable key across reruns regardless of
    # which sub-tab is currently in view. A single shared widget set that's
    # conditionally created only for whichever organ a radio button last
    # selected was tried first and dropped: Streamlit (and this app's own
    # test suite, which uses Streamlit's AppTest harness) does not
    # guarantee a orphaned widget key from a no-longer-rendered organ stays
    # harmless across every rerun path, and the tabs pattern below sidesteps
    # the question entirely rather than relying on that.
    single_organ_tabs = st.tabs([o.capitalize() for o in _ORGAN_ORDER])

    for organ_subtab, organ in zip(single_organ_tabs, _ORGAN_ORDER):
        with organ_subtab:
            organ_art = ORGAN_ARTIFACTS[organ]
            op = st.session_state.organ_params[organ]
            oc = st.session_state.organ_context[organ]

            if organ in _ORGAN_KNOWN_LIMITATIONS:
                st.info(_ORGAN_KNOWN_LIMITATIONS[organ])

            left, right = st.columns([1, 2])
            with left:
                if st.button("Reset %s to defaults" % organ, use_container_width=True, key="reset_organ_%s" % organ):
                    st.session_state.organ_params[organ] = {
                        **copy.deepcopy(organ_art["constructor_params"]),
                        **copy.deepcopy(organ_art["protocol_params"]),
                    }
                    st.session_state.organ_context[organ] = {
                        k: v["default"] for k, v in organ_art["context_schema"].items()
                    }
                    st.rerun()

                if oc:
                    st.markdown("**Context** — held fixed, normally supplied by another organ")
                    for name, meta in organ_art["context_schema"].items():
                        oc[name] = st.slider(
                            "%s (%s)" % (name, meta["unit"]), min_value=float(meta["min"]),
                            max_value=float(meta["max"]), value=float(oc[name]), step=float(meta["step"]),
                            key="octx_%s_%s" % (organ, name), help=meta["help"],
                        )

                with st.expander("Intrinsic parameters", expanded=False):
                    for name, meta in organ_art["param_schema"].items():
                        if name == "BUN_initial":
                            pinned = st.checkbox(
                                "Pin starting BUN instead of solving its steady state",
                                value=op[name] is not None, key="opin_%s" % organ,
                                help=meta["help"],
                            )
                            if pinned:
                                default_slider = op[name] if op[name] is not None else 12.0
                                op[name] = st.slider(
                                    "%s (%s)" % (name, meta["unit"]), min_value=float(meta["min"]),
                                    max_value=float(meta["max"]), value=float(default_slider),
                                    step=float(meta["step"]), key="octrl_%s_%s" % (organ, name),
                                    help=meta["help"],
                                )
                            else:
                                op[name] = None
                            continue
                        label = "%s (%s)" % (name, meta["unit"]) if meta.get("unit", "-") != "-" else name
                        current = float(np.clip(op[name], meta["min"], meta["max"]))
                        op[name] = st.slider(
                            label, min_value=float(meta["min"]), max_value=float(meta["max"]),
                            value=current, step=float(meta["step"]), key="octrl_%s_%s" % (organ, name),
                            help=meta["help"],
                        )

                organ_minutes = st.number_input(
                    "Duration (%s)" % ("seconds" if organ == "heart" else "minutes"),
                    min_value=(1.0 if organ == "heart" else 10.0),
                    max_value=(60.0 if organ == "heart" else 4320.0),
                    value=(10.0 if organ == "heart" else 1440.0),
                    step=(1.0 if organ == "heart" else 60.0),
                    key="ominutes_%s" % organ,
                )

                run_organ_clicked = st.button(
                    "▶ Run %s alone" % organ, type="primary", use_container_width=True,
                    key="orun_%s" % organ,
                )

            if run_organ_clicked:
                constructor_keys = list(organ_art["constructor_params"].keys())
                cp = {k: op[k] for k in constructor_keys}
                pp = {k: v for k, v in op.items() if k not in constructor_keys}
                with st.spinner("Integrating the %s alone..." % organ):
                    if organ == "heart":
                        odf, ometa = organ_runner.run_heart(cp, seconds=organ_minutes)
                    elif organ == "lungs":
                        odf, ometa = organ_runner.run_lungs(cp, pp, minutes=organ_minutes)
                    elif organ == "liver":
                        odf, ometa = organ_runner.run_liver(
                            cp, pp, test_glucose_mg_dL=oc["test_glucose_mg_dL"], minutes=organ_minutes)
                    elif organ == "kidney":
                        odf, ometa = organ_runner.run_kidney(
                            cp, pp, test_MAP_mmHg=oc["test_MAP_mmHg"], minutes=organ_minutes)
                    else:
                        odf, ometa = organ_runner.run_pancreas(cp, pp, minutes=organ_minutes)
                st.session_state.organ_result[organ] = {"df": odf, "meta": ometa}

            oresult = st.session_state.organ_result.get(organ)
            with right:
                if oresult is None:
                    st.info("Set parameters, then click Run %s alone." % organ)
                else:
                    odf, ometa = oresult["df"], oresult["meta"]
                    time_col, time_unit = _ORGAN_TIME_UNIT[organ]
                    st.subheader("Result — %d samples, %.0f %s" % (len(odf), odf[time_col].max(), time_unit))
                    st.caption(ometa["note"])
                    if ometa["held_fixed"]:
                        st.caption("Held fixed: " + ", ".join(
                            "%s = %s" % (k, v) for k, v in ometa["held_fixed"].items()))
                    if ometa["derived"]:
                        st.dataframe(
                            pd.DataFrame([ometa["derived"]]).round(3), use_container_width=True, hide_index=True,
                        )

                    panels = _ORGAN_PANELS[organ]
                    n_cols = 2
                    n_rows = int(np.ceil(len(panels) / n_cols))
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.0 * n_rows), sharex=True)
                    axes = np.atleast_1d(axes).ravel()
                    for ax, (col, label) in zip(axes, panels):
                        ax.plot(odf[time_col], odf[col], color=COLORS.get(col, "#374151"), linewidth=1.4)
                        ax.set_ylabel(label, fontsize=9)
                        ax.set_xlabel("Time (%s)" % time_unit, fontsize=8)
                        # sharex=True hides tick numbers on every row but the
                        # bottom one by default -- force them back on for
                        # every panel.
                        ax.tick_params(labelsize=8, labelbottom=True)
                    for ax in axes[len(panels):]:
                        ax.axis("off")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    st.download_button(
                        "Download this run as CSV",
                        data=odf.to_csv(index=False).encode("utf-8"),
                        file_name="%s_alone.csv" % organ,
                        mime="text/csv",
                        key="odl_%s" % organ,
                    )
