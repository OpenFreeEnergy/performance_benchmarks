import pathlib
import json

glob_search = pathlib.Path("ross_2023/").glob("*")
systems = [p for p in glob_search if p.is_dir()]


for system in systems:
    
    targets = [p for p in system.glob("*") if p.is_dir()]
    payload = {}
    
    for target in targets:
        metadata = {}
        name = target.name
        metadata["edge"] = str(target / f"{name}_edge.json")
        metadata["protein"] = str(target / "protein.pdb")
        
        cofactors = target / 'cofactors.sdf'
        if cofactors.exists():
            metadata["cofactors"] = str(cofactors)
        
        with open(target / "waters_dodecahedron_fast.json", "r") as fd:
            system_details = json.load(fd)
        
        metadata["atoms"] = {
            "complex": system_details["complex"][1],
            "solvent": system_details["solvent"][1],
        }
    
        metadata["waters"] = {
            "complex": system_details["complex"][0],
            "solvent": system_details["solvent"][0],
        }
        
        if name in payload:
            payload[f"{name}_1"] = metadata
        else:
            payload[name] = metadata
    
    with open(f"ross_dodecahedron_{system.name}.json", "w") as fd:
        json.dump(payload, fd, indent=4)
