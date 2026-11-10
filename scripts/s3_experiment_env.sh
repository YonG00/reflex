# Source after ROOT is assigned. Independent flags are validated, never silently cleared.
case "${GROUP:?GROUP required}" in
 S3) ;;
 A) export BOX_RETAIN_X10=1 ;;
 B) export BOX_DYNAMIC_BUFFER=1 ;;
 C) export BOX_FLEX_ARMS=1 ;;
 *) echo 'group must be S3, A, B or C' >&2; exit 2 ;;
esac
export BOX_BUFFER_STAGE=3 BOX_ESCAPE_PENALTY=0 BOX_IMPACT_BUFFER=0
PYTHONPATH="$ROOT/source/reflex${PYTHONPATH:+:$PYTHONPATH}" ${ISAACLAB_PY:-python} -m reflex.stage3_experiments "$GROUP"
