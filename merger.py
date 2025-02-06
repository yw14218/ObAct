import os
from pathlib import Path
import mujoco
from xml.etree import ElementTree as ET

class XMLMerger:
    def __init__(self, base_xml_path):
        self.base_xml_path = base_xml_path

    def merge_xmls(self, xml_paths):
        # Example: merge logic (placeholder)
        # Reads and combines XML files (actual logic depends on your requirements)
        base_tree = ET.parse(self.base_xml_path)
        base_root = base_tree.getroot()
        
        for path in xml_paths:
            tree = ET.parse(path)
            root = tree.getroot()
            for elem in root:
                base_root.append(elem)
        
        return ET.tostring(base_root, encoding='unicode')

def main():
    # Get the absolute path to the directory containing the script
    here = Path(__file__).resolve().parent

    # Use absolute paths for XML files
    scene_xml = here / "aloha" / "scene.xml"
    object_xml = here / "aloha" / "objects" / "can.xml"

    # Validate that XML files exist
    if not scene_xml.exists() or not object_xml.exists():
        raise FileNotFoundError("One or more XML files are missing.")

    # Create XML merger and merge XMLs
    xml_merger = XMLMerger(scene_xml)
    combined_xml = xml_merger.merge_xmls([object_xml])

    # Save or use the combined XML (for demonstration, let's save it)
    output_xml = here / "merged_scene.xml"
    with open(output_xml, "w") as f:
        f.write(combined_xml)

    # Load the Mujoco model and data
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    print("Mujoco model loaded successfully!")

if __name__ == "__main__":
    main()
