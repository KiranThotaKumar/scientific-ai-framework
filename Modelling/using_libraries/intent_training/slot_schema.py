
#using_libraries\intent_training\slot_schema.py

SCHEMA_VERSION = "v1"


DOMAIN_LABELS = ["hydrogen", "single_qubit", "multi_qubit"]

ACTION_LABELS = ["forward", "infer_parameters"]

SERIES_LABELS = ["none", "Lyman", "Balmer", "Paschen", "Brackett", "Pfund"]

SPECTRUM_MODE_LABELS = ["none", "absorption", "emission"]

EVOLUTION_MODE_LABELS = ["none", "time", "frequency"]

OPEN_SYSTEM_MODE_LABELS = {
    "none": 0,
    "closed": 1,
    "open": 2
}

COUPLING_TOPOLOGY_LABELS = {
    "none": 0,
    "nearest_neighbor": 1,
    "all_to_all": 2,
    "chain": 3,
    "ring": 4
}

INITIAL_STATE_FAMILY_LABELS = {
    "none": 0,
    "ground": 1,
    "excited": 2,
    "superposition": 3,
    "bell": 4,
    "ghz": 5
}

OBSERVABLE_TYPE_LABELS = {
    "none": 0,
    "population": 1,
    "coherence": 2,
    "pauli_z": 3,
    "pauli_x": 4
}

CONTINUOUS_SLOTS_V2 = [

    # --- Hydrogen Forward ---
    "emin",
    "emax",
    "sigma_instr",
    "background",
    "scale",

    # --- Single Qubit ---
    "omega_r",
    "detuning",
    "gamma",
    "tmax",
    "amplitude",
    "offset",

    # --- Multi Qubit ---
    "coupling_strength",
    "noise_std",
    #"n_qubits",
]
# =========================
# Domain registry
# =========================

DOMAIN2ID = {
    "hydrogen": 0,
    "single_qubit": 1,
    "multi_qubit": 2,
}

ID2DOMAIN = {v: k for k, v in DOMAIN2ID.items()}

HYDROGEN_DOMAIN_ID = DOMAIN2ID["hydrogen"]
SINGLE_QUBIT_DOMAIN_ID = DOMAIN2ID["single_qubit"]
MULTI_QUBIT_DOMAIN_ID = DOMAIN2ID["multi_qubit"]

QUBIT_DOMAIN_IDS = {
    SINGLE_QUBIT_DOMAIN_ID,
    MULTI_QUBIT_DOMAIN_ID,
}

DOMAIN_MAP = {
    "hydrogen": 0,
    "single_qubit": 1,
    "multi_qubit": 2
}

ACTION_MAP = {
    "forward": 0,
    "infer_parameters": 1
}

CATEGORICAL_SLOT_VOCABULARIES = {
    "series": ["none", "Lyman", "Balmer", "Paschen", "Pfund"],
    "spectrum_mode": ["none", "absorption", "emission"],
    "evolution_mode": ["none", "time", "frequency"],    

    "open_system_mode": [
        "none",
        "closed",
        "open"
    ],

    "coupling_topology": [
        "none",
        "nearest_neighbor",
        "all_to_all",
        "chain",
        "ring"
    ],

    "initial_state_family": [
        "none",
        "ground",
        "excited",
        "superposition",
        "bell",
        "ghz"
    ],

    "observable_type": [
        "none",
        "population",
        "coherence",
        "pauli_z",
        "pauli_x"
    ]

}

CONT_SLOT_INDEX = {
    name: idx for idx, name in enumerate(CONTINUOUS_SLOTS_V2)
}

SLOT_MASKS_V2 = {

    ("hydrogen", "forward"): {
        "categorical": ["series", "spectrum_mode"],
        "continuous": ["emin", "emax", "sigma_instr", "background", "scale"]
    },

    ("hydrogen", "infer_parameters"): {
        "categorical": ["file_name"],
        "continuous": []
    },

    ("single_qubit", "forward"): {
        "categorical": [],
        "continuous": [
            "omega_r",
            "detuning",
            "gamma",
            "amplitude",
            "offset",
            "tmax"
        ]
    },

    ("single_qubit", "infer_parameters"): {
        "categorical": ["file_name"],
        "continuous": []
    },

    ("multi_qubit", "forward"): {

        "categorical": [
            "evolution_mode",
            "open_system_mode",
            "coupling_topology",
            "initial_state_family",
            "observable_type"
        ],

        "continuous": [
            "coupling_strength",
            "omega_r",
            "gamma",
            "detuning",
            "noise_std",
            "sigma_instr",
            "background",
            "tmax"
        ]
    },

    ("multi_qubit", "infer_parameters"): {
        "categorical": ["file_name"],
        "continuous": []
    }
}


DOMAIN2ID = {
    "hydrogen": 0,
    "single_qubit": 1,
    "multi_qubit": 2,
}

ACTION2ID = {
    "forward": 0,
    "infer_parameters": 1,
}

REGRESSION_SLOT_ORDER = [
    "emin",
    "emax",
    "sigma_instr",
    "background",
    "scale",
    "omega_r",
    "detuning",
    "gamma",
    "tmax",
    "amplitude",
    "offset",
    "coupling_strength",
    "noise_std",
    #"n_qubits",
]

SERIES2ID = {
    label: idx
    for idx, label in enumerate(SERIES_LABELS)
}

ID2SERIES = {
    idx: label
    for label, idx in SERIES2ID.items()
}

SPECTRUM_MODE2ID = {
    "absorption": 0,
    "emission": 1,
}

EVOLUTION_MODE2ID = {
    "none": 0,
    "time": 1,
    "frequency": 2
}

ID2EVOLUTION_MODE = {
    0: "none",
    1: "time",
    2: "frequency"
}


NUM_DOMAINS = len(DOMAIN2ID)
NUM_ACTIONS = len(ACTION2ID)
NUM_SPECTRUM_MODES = len(SPECTRUM_MODE2ID)
NUM_SERIES_LABELS = len(SERIES2ID)
NUM_EVOLUTION_MODES = len(EVOLUTION_MODE2ID)

NUM_CONTINUOUS = len(CONTINUOUS_SLOTS_V2)  # should be 12

NUM_OPEN_SYSTEM_MODES = len(OPEN_SYSTEM_MODE_LABELS)
NUM_COUPLING_TOPOLOGIES = len(COUPLING_TOPOLOGY_LABELS)
NUM_INITIAL_STATE_FAMILIES = len(INITIAL_STATE_FAMILY_LABELS)
NUM_OBSERVABLE_TYPES = len(OBSERVABLE_TYPE_LABELS)

MODEL_CONFIG = {

    "num_domains": NUM_DOMAINS,
    "num_actions": NUM_ACTIONS,

    "num_spectrum_modes": NUM_SPECTRUM_MODES,
    "num_series_labels": NUM_SERIES_LABELS,
    "num_evolution_modes": NUM_EVOLUTION_MODES,

    "num_open_system_modes": NUM_OPEN_SYSTEM_MODES,
    "num_coupling_topologies": NUM_COUPLING_TOPOLOGIES,
    "num_initial_state_families": NUM_INITIAL_STATE_FAMILIES,
    "num_observable_types": NUM_OBSERVABLE_TYPES,

    "num_continuous": NUM_CONTINUOUS,
    "max_length": 128
}

DOMAIN_ACTION_MODES_CONFIG = {

    "hydrogen": {

        "forward": {

            "series": True,

            "spectrum_mode": True,

            "evolution_mode": False,

            "open_system_mode": False,

            "coupling_topology": False,

            "initial_state_family": False,

            "observable_type": False
        },

        "infer_parameters": {

            "series": False,

            "spectrum_mode": False,

            "evolution_mode": False,

            "open_system_mode": False,

            "coupling_topology": False,

            "initial_state_family": False,

            "observable_type": False
        }
    },

    "single_qubit": {

        "forward": {

            "series": False,

            "spectrum_mode": False,

            "evolution_mode": True,

            "open_system_mode": True,

            "coupling_topology": False,

            "initial_state_family": False,

            "observable_type": True
        },

        "infer_parameters": {

            "series": False,

            "spectrum_mode": False,

            "evolution_mode": False,

            "open_system_mode": False,

            "coupling_topology": False,

            "initial_state_family": False,

            "observable_type": False
        }
    },

    "multi_qubit": {

        "forward": {

            "series": False,

            "spectrum_mode": False,

            "evolution_mode": True,

            "open_system_mode": True,

            "coupling_topology": True,

            "initial_state_family": True,

            "observable_type": True
        },

        "infer_parameters": {

            "series": False,

            "spectrum_mode": False,

            "evolution_mode": False,

            "open_system_mode": False,

            "coupling_topology": False,

            "initial_state_family": False,

            "observable_type": False
        }
    }
}