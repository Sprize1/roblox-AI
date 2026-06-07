#!/usr/bin/env python3
"""Generate a Luau script that clears the workspace garbage then injects the diffusion layouts
(generated row at Z=0 blue, real row at Z=300 green), side by side for visual comparison."""
import json

d = json.load(open("data/diff_gen.json"))
L = ["-- clear garbage (keep Terrain/Camera)",
     "for _,c in ipairs(workspace:GetChildren()) do",
     "  if not (c:IsA('Terrain') or c:IsA('Camera')) then pcall(function() c:Destroy() end) end",
     "end",
     "local root = Instance.new('Model'); root.Name='DiffViz_3D'; root.Parent=workspace",
     "local function box(parent,x,y,z,sx,sy,sz,r,g,b)",
     "  local p=Instance.new('Part'); p.Anchored=true; p.Size=Vector3.new(sx,sy,sz)",
     "  p.Position=Vector3.new(x,y,z); p.Color=Color3.fromRGB(r,g,b); p.Parent=parent",
     "end"]

def emit(tag, layouts, offz, rgb):
    L.append(f"local grp_{tag}=Instance.new('Model'); grp_{tag}.Name='{tag}'; grp_{tag}.Parent=root")
    for i, lay in enumerate(layouts):
        offx = i * 130
        for b in lay:
            x, y, z, sx, sy, sz = b
            L.append(f"box(grp_{tag},{x+offx:.1f},{y:.1f},{z+offz:.1f},{sx:.1f},{sy:.1f},{sz:.1f},{rgb[0]},{rgb[1]},{rgb[2]})")

emit("Generated", d["generated"], 0, (90, 150, 255))
emit("Real", d["real"], 300, (90, 220, 130))
L.append("return 'DiffViz_3D: '..#root.Generated:GetChildren()..' gen parts, '..#root.Real:GetChildren()..' real parts'")

open("models/inject.luau", "w").write("\n".join(L))
print(f"wrote models/inject.luau ({len(L)} lines, {sum(len(x) for x in d['generated'])+sum(len(x) for x in d['real'])} parts)")
