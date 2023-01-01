for scene in "$@"
do
    echo "$scene"
    python scripts/colmap2nerf.py --images ./data/toybox-5/$scene/Images/ --run_colmap
done


