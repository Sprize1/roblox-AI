#!/usr/bin/env python3
"""Emit Luau to clear workspace + inject whole-game generations (blue, Z=0) vs real games
(green, Z=1100), each game offset in X for side-by-side viewing."""
import json

d = json.load(open("data/whole_gen.json"))
L = ["for _,c in ipairs(workspace:GetChildren()) do",
     "  if not (c:IsA('Terrain') or c:IsA('Camera')) then pcall(function() c:Destroy() end) end end",
     "local root=Instance.new('Model'); root.Name='WholeViz'; root.Parent=workspace",
     "local function box(p,x,y,z,sx,sy,sz,r,g,b)",
     " local q=Instance.new('Part'); q.Anchored=true; q.Size=Vector3.new(sx,sy,sz)",
     " q.Position=Vector3.new(x,y,z); q.Color=Color3.fromRGB(r,g,b); q.Parent=p end"]

def emit(tag, games, offz, rgb, n):
    grp = f"g_{tag}"
    L.append(f"local {grp}=Instance.new('Model'); {grp}.Name='{tag}'; {grp}.Parent=root")
    for i, game in enumerate(games[:n]):
        ox = i * 700
        for b in game:
            x, y, z, sx, sy, sz = b
            L.append(f"box({grp},{x+ox:.1f},{y:.1f},{z+offz:.1f},"
                     f"{max(1,abs(sx)):.1f},{max(1,abs(sy)):.1f},{max(1,abs(sz)):.1f},{rgb[0]},{rgb[1]},{rgb[2]})")

emit("Generated", d["generated"], 0, (90, 150, 255), 2)
emit("Real", d["real"], 1100, (90, 220, 130), 1)
L.append("return #g_Generated:GetDescendants()..' gen, '..#g_Real:GetDescendants()..' real parts'")
open("models/whole_inject.luau", "w").write("\n".join(L))
print(f"wrote models/whole_inject.luau ({len(L)} lines)")
