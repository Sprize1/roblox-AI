#!/usr/bin/env python3
"""Emit a compact Luau (flat number table + loop) to inject one full generated whole game."""
import json

d = json.load(open("data/whole_gen.json"))
g = d["generated"][0][::4]                              # 1/4 stride subsample (~250 parts) to fit plugin payload
flat = []
for b in g:
    flat += [round(b[0], 1), round(b[1], 1), round(b[2], 1),
             round(max(1.0, abs(b[3])), 1), round(max(1.0, abs(b[4])), 1), round(max(1.0, abs(b[5])), 1)]
data = ",".join(str(v) for v in flat)

L = [
    "for _,c in ipairs(workspace:GetChildren()) do if not (c:IsA('Terrain') or c:IsA('Camera')) then pcall(function() c:Destroy() end) end end",
    "local root=Instance.new('Model'); root.Name='WholeGenViz'; root.Parent=workspace",
    "local D={" + data + "}",
    "for i=1,#D,6 do",
    " local p=Instance.new('Part'); p.Anchored=true",
    " p.Size=Vector3.new(D[i+3],D[i+4],D[i+5]); p.Position=Vector3.new(D[i],D[i+1],D[i+2])",
    " p.Color=Color3.fromRGB(90,150,255); p.Parent=root end",
    "return #root:GetChildren()..' parts'",
]
open("models/compact_inject.luau", "w").write("\n".join(L))
print(f"wrote models/compact_inject.luau ({len(g)} parts, ~{len(data)//1000}KB data)")
