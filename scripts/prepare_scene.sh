FOLDER=/home/leo/torch-ngp/data/toybox-5/$1
ORIGINAL_FOLDER=$PWD
cd $FOLDER

mkdir Images
mkdir Labels
rm -rf depth*
mv rgba* Images/
mv segmentation* Labels/
cd Images
for file in *; do mv "$file" "${file//rgba_}"; done;
cd ../Labels
for file in *; do mv "$file" "${file//segmentation_}"; done;

cd $ORIGINAL_FOLDER
