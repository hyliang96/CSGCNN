import scipy.io

# Load annotations from .mat files creating a Python dictionary:
def load_annotations(path):

    # Get annotations from the file and relative objects:
    annotations = scipy.io.loadmat(path)["anno"]

    objects = annotations[0, 0]["objects"]

    # List containing information of each object (to add to dictionary):
    objects_list = []

    # Go through the objects and extract info:
    for obj_idx in range(objects.shape[1]):
        obj = objects[0, obj_idx]

        # Get classname and mask of the current object:
        classname = obj["class"][0]
        mask = obj["mask"]

        # List containing information of each body part (to add to dictionary):
        parts_list = []

        parts = obj["parts"]

        # Go through the part of the specific object and extract info:
        for part_idx in range(parts.shape[1]):
            part = parts[0, part_idx]
            # Get part name and mask of the current body part:
            part_name = part["part_name"][0]
            part_mask = part["mask"]

            # Add info to parts_list:
            parts_list.append({"part_name": part_name, "mask": part_mask})

        # Add info to objects_list:
        objects_list.append({"class": classname, "mask": mask, "parts": parts_list})

    return {"objects": objects_list}
