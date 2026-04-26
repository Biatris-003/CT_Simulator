import os
import re

directories = [
    "extracted_BeamGeometry_v_1_4", 
    "extracted_ImagingParameters_v_1_4", 
    "extracted_ReconstructionAlgo_v_1_4", 
    "extracted_simulatorCTlab_1_4", 
    "extracted_spectraTool_v_1_4", 
    "extracted_spectralParameters_v_1_4"
]

for d in directories:
    xml_path = f"j:/CTlab/{d}/matlab/document.xml"
    if os.path.exists(xml_path):
        with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        matches = re.findall(r"<!\[CDATA\[(.*?)\]\]>", content, re.DOTALL)
        if matches:
            m_code = matches[0]
            out_path = f"j:/CTlab/{d.replace('extracted_', '')}.m"
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(m_code)
            print(f"Extracted {out_path}")
        else:
            print(f"No CDATA found in {xml_path}")
