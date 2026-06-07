"""Proper Roblox XML parser. Streaming, robust, handles all .rbxlx/.rbxmx formats."""
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

def _safe_float(t, d=0.0):
    """float() that never raises (modern composites mix bools like <CustomPhysics>false</CustomPhysics>)."""
    try: return float(t)
    except (TypeError, ValueError): return d

@dataclass
class RobloxInstance:
    class_name: str
    referent: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    parent_ref: Optional[str] = None
    children: List[str] = field(default_factory=list)

def parse_property_value(tag: str, text: str) -> Any:
    """Parse a Roblox property value from XML tag type and text content."""
    text = text.strip() if text else ""
    tag = tag.lower()

    if tag in ("string", "protectedstring", "content", "binarystring"):
        return text
    elif tag == "bool":
        return text.lower() == "true"
    elif tag in ("int", "int64"):
        return int(text) if text.lstrip("-").isdigit() else 0
    elif tag in ("float", "double", "number"):
        try: return float(text)
        except: return 0.0
    elif tag in ("vector3", "vector3int16"):
        parts = [float(x.strip()) for x in text.replace(",", " ").split() if x.strip()]
        return {'x': parts[0] if len(parts)>0 else 0,
                'y': parts[1] if len(parts)>1 else 0,
                'z': parts[2] if len(parts)>2 else 0}
    elif tag == "vector2":
        parts = [float(x.strip()) for x in text.replace(",", " ").split() if x.strip()]
        return {'x': parts[0] if len(parts)>0 else 0, 'y': parts[1] if len(parts)>1 else 0}
    elif tag == "color3uint8":
        parts = [float(x.strip())/255 for x in text.replace(",", " ").split() if x.strip()]
        return (parts[0] if len(parts)>0 else 0.5,
                parts[1] if len(parts)>1 else 0.5,
                parts[2] if len(parts)>2 else 0.5)
    elif tag == "color3":
        parts = [float(x.strip()) for x in text.replace(",", " ").split() if x.strip()]
        return (parts[0] if len(parts)>0 else 1.0,
                parts[1] if len(parts)>1 else 1.0,
                parts[2] if len(parts)>2 else 1.0)
    elif tag == "cframe" or tag == "coordinateframe":
        # CFrame can be 12 numbers (matrix) or 3 numbers (position only)
        parts = [float(x.strip()) for x in text.replace(",", " ").split() if x.strip()]
        if len(parts) >= 3:
            return {'x': parts[0], 'y': parts[1], 'z': parts[2]}
        return {'x': 0, 'y': 0, 'z': 0}
    elif tag == "udim2":
        parts = [float(x.strip()) for x in text.replace(","," ").replace("{"," ").replace("}"," ").split() if x.strip()]
        return {'x': parts[0] if len(parts)>0 else 0, 'y': parts[1] if len(parts)>1 else 0}
    elif tag == "ref":
        return text
    elif tag == "token":
        return int(text) if text.isdigit() else 0
    elif tag == "brickcolor":
        return BRICKCOLOR_TO_RGB.get(int(text) if text.isdigit() else 0, (0.5,0.5,0.5))
    elif tag == "material":
        return text  # "Plastic", "Grass", "Concrete", etc.
    else:
        return text


# BrickColor index → RGB mapping (common colors)
BRICKCOLOR_TO_RGB = {
    1: (0.95, 0.95, 0.95),   # White
    101: (0.54, 0.16, 0.14),  # Bright red
    102: (0.18, 0.38, 0.18),  # Bright green
    104: (0.18, 0.18, 0.43),  # Bright blue
    192: (0.49, 0.35, 0.19),  # Brown
    194: (0.40, 0.40, 0.40),  # Dark stone grey
    199: (0.37, 0.37, 0.37),  # Dark grey
    1001: (1.0, 1.0, 1.0),    # Institutional white
    1002: (0.76, 0.76, 0.76), # Mid grey
    1010: (0.16, 0.20, 0.31), # Dark blue
    1011: (0.10, 0.39, 0.37), # Teal
    1012: (0.50, 0.50, 0.50), # Medium grey
    1013: (0.77, 0.77, 0.77), # Light grey
    1021: (0.98, 0.93, 0.45), # Bright yellow
    1024: (0.42, 0.59, 0.97), # Bright blue
}
# Default: medium grey for unknown indices
def _get_brick_color(idx): return BRICKCOLOR_TO_RGB.get(idx, (0.5, 0.5, 0.5))

# Material enum (real Enum.Material values) → name
MATERIAL_ENUM = {
    256: "Plastic", 272: "SmoothPlastic", 288: "Neon",
    512: "Wood", 528: "WoodPlanks",
    784: "Marble", 788: "Basalt", 800: "Slate", 804: "CrackedLava",
    816: "Concrete", 820: "Limestone", 832: "Granite", 836: "Pavement",
    848: "Brick", 864: "Pebble", 880: "Cobblestone", 890: "Mud",
    896: "Rock", 920: "Sandstone",
    1040: "CorrodedMetal", 1056: "DiamondPlate", 1072: "Foil", 1088: "Metal",
    1280: "Grass", 1284: "LeafyGrass", 1296: "Sand", 1312: "Fabric",
    1328: "Snow", 1344: "Ground", 1376: "Asphalt", 1392: "Salt",
    1536: "Ice", 1552: "Glacier", 1568: "Glass", 1584: "ForceField",
    1792: "Air", 2048: "Water",
}


def _assign_property(props, child):
    """Parse one XML property element (scalar or composite) into props.
    Shared by the streaming and fromstring parsers so both read rotation/composites identically."""
    pname = child.get('name', child.tag)
    tag_lower = child.tag.lower()
    if pname == "Material" and tag_lower == "token":
        t = (child.text or '').strip()
        mat_id = int(t) if t.lstrip('-').isdigit() else 0
        props["Material"] = MATERIAL_ENUM.get(mat_id, f"Unknown_{mat_id}")
        return
    if tag_lower in ('coordinateframe', 'cframe'):
        nums = {sub.tag: _safe_float(sub.text) for sub in child}
        val = type('CF', (), {'x': nums.get('X', 0), 'y': nums.get('Y', 0), 'z': nums.get('Z', 0)})
        val.R00 = nums.get('R00', 1); val.R01 = nums.get('R01', 0); val.R02 = nums.get('R02', 0)
        val.R10 = nums.get('R10', 0); val.R11 = nums.get('R11', 1); val.R12 = nums.get('R12', 0)
        val.R20 = nums.get('R20', 0); val.R21 = nums.get('R21', 0); val.R22 = nums.get('R22', 1)
        props[pname] = val
    elif tag_lower in ('vector3', 'vector3int16'):
        nums = {sub.tag: _safe_float(sub.text) for sub in child}
        props[pname] = type('V3', (), {'x': nums.get('X', 0), 'y': nums.get('Y', 0), 'z': nums.get('Z', 0)})()
    elif tag_lower == 'vector2':
        nums = {sub.tag: _safe_float(sub.text) for sub in child}
        props[pname] = type('V2', (), {'x': nums.get('X', 0), 'y': nums.get('Y', 0)})()
    elif tag_lower in ('color3', 'color3uint8'):
        nums = {sub.tag: _safe_float(sub.text) for sub in child}
        r, g, b = nums.get('R', 255), nums.get('G', 255), nums.get('B', 255)
        if tag_lower == 'color3uint8': r, g, b = r / 255, g / 255, b / 255
        props[pname] = (r, g, b)
    elif tag_lower in ('udim2', 'udim'):
        nums = {sub.tag: _safe_float(sub.text) for sub in child}
        props[pname] = type('U2', (), {'x': nums.get('XS', 0), 'y': nums.get('YS', 0)})()
    elif tag_lower == 'physicalproperties':
        props[pname] = str({sub.tag: _safe_float(sub.text) for sub in child})
    elif tag_lower in ('content', 'contentid'):
        url = ''
        for sub in child:
            if (sub.text or '').strip():
                url = sub.text.strip(); break
        props[pname] = url or (child.text or '').strip()
    else:
        props[pname] = parse_property_value(child.tag, child.text or "")


def parse_file(filepath: str) -> Dict[str, RobloxInstance]:
    """Streaming parser (memory-safe for huge files). Reads composite properties
    (CFrame with rotation, Vector3, Color3, …) from nested children — like fromstring."""
    instances: Dict[str, RobloxInstance] = {}
    stack: List[str] = []
    depth = 0
    props_depth = None
    try:
        for event, elem in ET.iterparse(filepath, events=('start', 'end')):
            tag = elem.tag.lower()
            if event == 'start':
                depth += 1
                if tag == 'item':
                    referent = elem.get('referent', str(len(instances)))
                    parent = stack[-1] if stack else None
                    instances[referent] = RobloxInstance(class_name=elem.get('class', 'Unknown'),
                                                         referent=referent, parent_ref=parent)
                    if parent and parent in instances:
                        instances[parent].children.append(referent)
                    stack.append(referent)
                elif tag == 'properties':
                    props_depth = depth
            else:  # end
                if tag == 'item':
                    if stack: stack.pop()
                    elem.clear()
                elif tag == 'properties':
                    props_depth = None
                elif props_depth is not None and depth == props_depth + 1:
                    if stack and stack[-1] in instances:    # a property element; its children are still attached at its end
                        _assign_property(instances[stack[-1]].properties, elem)
                    elem.clear()
                depth -= 1
    except ET.ParseError:
        try:
            return parse_file_fromstring(filepath)
        except Exception:
            pass
    return instances


def parse_file_fromstring(filepath: str) -> Dict[str, RobloxInstance]:
    """Fallback: parse entire file at once (faster for small files)."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Strip doctype if present
    if content.startswith('<!'):
        content = content[content.find('>')+1:]

    root = ET.fromstring(content)
    instances = {}

    def walk(elem, parent_ref=None):
        class_name = elem.get('class', 'Unknown')
        referent = elem.get('referent', str(len(instances)))
        props = {}

        for props_elem in elem.findall('Properties'):
            for child in props_elem:
                pname = child.get('name', child.tag)
                tag_lower = child.tag.lower()

                # Handle token Material specially
                if pname == "Material" and tag_lower == "token":
                    mat_id = int(child.text or 0)
                    props["Material"] = MATERIAL_ENUM.get(mat_id, f"Unknown_{mat_id}")
                    continue

                # Handle composite types
                if tag_lower in ('coordinateframe', 'cframe'):
                    nums = {}
                    for sub in child:
                        nums[sub.tag] = _safe_float(sub.text)
                    val = type('CF', (), {'x': nums.get('X', 0), 'y': nums.get('Y', 0), 'z': nums.get('Z', 0)})
                    val.R00 = nums.get('R00', 1); val.R01 = nums.get('R01', 0)
                    val.R02 = nums.get('R02', 0); val.R10 = nums.get('R10', 0)
                    val.R11 = nums.get('R11', 1); val.R12 = nums.get('R12', 0)
                    val.R20 = nums.get('R20', 0); val.R21 = nums.get('R21', 0)
                    val.R22 = nums.get('R22', 1)
                    props[pname] = val
                elif tag_lower in ('vector3', 'vector3int16'):
                    nums = {}
                    for sub in child:
                        nums[sub.tag] = _safe_float(sub.text)
                    props[pname] = type('V3', (), {'x': nums.get('X', 0), 'y': nums.get('Y', 0), 'z': nums.get('Z', 0)})()
                elif tag_lower == 'vector2':
                    nums = {}
                    for sub in child:
                        nums[sub.tag] = _safe_float(sub.text)
                    props[pname] = type('V2', (), {'x': nums.get('X', 0), 'y': nums.get('Y', 0)})()
                elif tag_lower in ('color3', 'color3uint8'):
                    nums = {}
                    for sub in child:
                        nums[sub.tag] = _safe_float(sub.text)
                    r, g, b = nums.get('R', 255), nums.get('G', 255), nums.get('B', 255)
                    if tag_lower == 'color3uint8':
                        r, g, b = r/255, g/255, b/255
                    props[pname] = (r, g, b)
                elif tag_lower in ('udim2', 'udim'):
                    nums = {}
                    for sub in child:
                        nums[sub.tag] = _safe_float(sub.text)
                    props[pname] = type('U2', (), {'x': nums.get('XS', 0), 'y': nums.get('YS', 0)})()
                elif tag_lower == 'physicalproperties':
                    nums = {}
                    for sub in child:
                        nums[sub.tag] = _safe_float(sub.text) if sub.text else 0
                    props[pname] = str(nums)
                elif tag_lower in ('content', 'contentid'):
                    # Content (MeshId, TextureId, …) wraps a nested <url>/<hash> child;
                    # the value is not the element's own text. Read the first non-empty child.
                    url = ''
                    for sub in child:
                        if (sub.text or '').strip():
                            url = sub.text.strip(); break
                    props[pname] = url or (child.text or '').strip()
                else:
                    ptext = child.text or ""
                    value = parse_property_value(child.tag, ptext)
                    props[pname] = value

        inst = RobloxInstance(class_name=class_name, referent=referent,
                              properties=props, parent_ref=parent_ref)
        instances[referent] = inst

        for child_elem in elem.findall('Item'):
            walk(child_elem, referent)

    for item in root.findall('Item'):
        walk(item)

    return instances


def parse_string(xml_string: str) -> Dict[str, RobloxInstance]:
    """Parse from string (for binary files decompressed to XML)."""
    instances = {}

    try:
        root = ET.fromstring(xml_string)

        def walk(elem, parent_ref=None):
            cn = elem.get('class', 'Unknown')
            ref = elem.get('referent', str(len(instances)))
            props = {}

            for props_elem in elem.findall('Properties'):
                for child in props_elem:
                    pname = child.get('name', child.tag)
                    ptext = child.text or ""
                    props[pname] = parse_property_value(child.tag, ptext)

            inst = RobloxInstance(class_name=cn, referent=ref,
                                  properties=props, parent_ref=parent_ref)
            instances[ref] = inst

            for child_elem in elem.findall('Item'):
                walk(child_elem, ref)

        # Handle wrapper elements
        items = root.findall('Item')
        if not items:
            items = root.findall('.//Item')[:1]  # Try deeper

        for item in items:
            walk(item)

    except ET.ParseError:
        return {}

    return instances
