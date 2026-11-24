#!/usr/bin/env bash

# Pass OBSID as first argument, or fallback to default
OBSID="${1:-00050300004}"

# Path definitions
BASE_DIR="/Users/tarnesen/3Dswift/notebooks/3D-XRT_tutorials/data"
TARGET_DIR="${BASE_DIR}/reproc/${OBSID}/xrt/mine"
EVENT_DIR="${BASE_DIR}/reproc/${OBSID}/xrt/event"
REF_MINE_DIR="${BASE_DIR}/reproc/00050300004/xrt/mine"

# File definitions
REG_FILE="all_602.reg"
EVT_FILE="sw${OBSID}xpcw4po_cl.evt.gz"
SESSION_NAME="${OBSID}_arf_602"
OUT_SPEC="PC_all_602.pi"
OUT_ARF="Obs_${OBSID}pc_all_602.arf"

# 1. Create target directory if missing
mkdir -p "${TARGET_DIR}"

# 2. Copy region file from reference directory
if [ -f "${REF_MINE_DIR}/${REG_FILE}" ]; then
    cp "${REF_MINE_DIR}/${REG_FILE}" "${TARGET_DIR}/"
else
    echo "Warning: Region file not found at ${REF_MINE_DIR}/${REG_FILE}"
fi

# 3. Copy event files to mine directory
cp "${EVENT_DIR}"/sw* "${TARGET_DIR}/"

# 4. Change into target directory so XSELECT runs locally
cd "${TARGET_DIR}"

# 5. Run XSELECT interactive workflow
xselect << EOF
${SESSION_NAME}
no
read event
${EVENT_DIR}
${EVT_FILE}
yes
filter region ${REG_FILE}
extract spectrum
save spectrum ${OUT_SPEC}
yes
quit
yes
EOF

# 6. Delete already existing arf
rm "${OUT_ARF}"

# 7. Run XRTMKARF interactive workflow
xrtmkarf << EOF
NONE
${OUT_SPEC}
no
${OUT_ARF}
-1
-1
EOF