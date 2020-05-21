## Expert sparse labels

[labels_full.csv](./labels_full.csv) contains all the manual labels used for the project.

It contains one row per label, each containing the following values:
* `X`, `Y` and `Z` representing the voxel position. 
**NOTE:** This are *1-based* for use in matlab. When used in python, subtract 1 from each.
* `label` is a binary label, `1` for vessel voxels, and `0` for non-vessels.
* `scanID` is the three-character ID of the scan the labels are for.

The Scan IDs correspond to identifiers within the MIDAS database, containing TOF-MRA scans of healthy patients:  
https://www.insight-journal.org/midas/community/view/21 (Bullitt, Smith, and Lin). For example, a label for '002' corresponds to a voxel within the Normal002-MRA volume.
