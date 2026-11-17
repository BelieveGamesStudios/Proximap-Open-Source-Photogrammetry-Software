"""Import preparation and base-colour texture assignment without GL work."""
from pathlib import Path
import re
import shlex
import numpy as np
from PIL import Image
from mesh_editor.scene import Object

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tga', '.webp', '.tif', '.tiff'}


def texture_candidates(paths):
    """Return images and explicit MTL material-to-image links."""
    images, links = [], {}
    for value in paths:
        path = Path(value)
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            images.append(str(path))
        elif path.suffix.lower() == '.mtl' and path.is_file():
            material = None
            for line in path.read_text(errors='replace').splitlines():
                try:
                    lexer = shlex.shlex(line, posix=True)
                    lexer.whitespace_split = True
                    lexer.commenters = '#'
                    lexer.escape = ''
                    parts = list(lexer)
                except ValueError:
                    continue
                if not parts:
                    continue
                if parts[0] == 'newmtl':
                    material = ' '.join(parts[1:])
                elif parts[0] == 'map_Kd' and material:
                    # Try the complete remainder first (unquoted spaces), then
                    # trailing arguments for MTL map options such as -s/-o.
                    args = parts[1:]
                    candidates = [' '.join(args)] + [' '.join(args[i:]) for i in range(1, len(args))]
                    for name in candidates:
                        image = path.parent / name.replace('\\', '/')
                        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                            links[material.casefold()] = str(image)
                            images.append(str(image))
                            break
    return list(dict.fromkeys(images)), links


def _key(name):
    value = re.sub(r'[^a-z0-9]', '', str(name).casefold())
    return re.sub(r'(basecolor|basecolour|diffuse|albedo|color|colour)$', '', value)


def match_textures(meshes, paths):
    images, links = texture_candidates(paths)
    materials = list(dict.fromkeys(mesh.material_name for _, mesh in meshes))
    assignments = {}
    for material in materials:
        if material.casefold() in links:
            assignments[material] = links[material.casefold()]
            continue
        keys = {_key(material)}
        keys.update(_key(name) for name, mesh in meshes if mesh.material_name == material and name)
        matches = [path for path in images if _key(Path(path).stem) in keys]
        if len(matches) == 1:
            assignments[material] = matches[0]
        elif len(materials) == 1:
            base_images = [path for path in images if re.search(r'(base[_ -]?colou?r|diffuse|albedo)', Path(path).stem, re.I)]
            if len(base_images) == 1:
                assignments[material] = base_images[0]
            elif len(images) == 1 and not re.search(r'(normal|roughness|metallic|occlusion|height|bump)', Path(images[0]).stem, re.I):
                assignments[material] = images[0]
    return images, assignments


def apply_assignments(meshes, assignments):
    """Only override explicitly assigned materials; unassigned ones use colour."""
    cache = {}
    for _, mesh in meshes:
        mesh.texture_data = None
        path = assignments.get(mesh.material_name)
        if not path:
            continue
        if mesh.texcoords is None:
            raise ValueError(f"Material '{mesh.material_name}' has no UV coordinates for an external texture.")
        if path not in cache:
            with Image.open(path) as source:
                cache[path] = source.convert('RGBA').copy()
        mesh.texture_data = cache[path]


def selection_roots(objects):
    selected = set(objects)
    return [obj for obj in objects if not any(parent in selected for parent in obj.ancestors())]


def group_objects(objects, name):
    roots = selection_roots(objects)
    if not roots:
        raise ValueError('Select at least one object to parent.')
    bounds = [obj.get_world_aabb() for obj in roots]
    center = (np.min([b[0] for b in bounds], axis=0) + np.max([b[1] for b in bounds], axis=0)) / 2
    parent = Object(name, None)
    parent.position = center.astype(np.float32)
    common_parent = roots[0].parent
    if common_parent is not None and all(obj.parent is common_parent for obj in roots):
        parent.set_parent(common_parent)
    for obj in roots:
        obj.set_parent(parent)
    return parent


def build_import_objects(meshes, file_path, group=True, existing_names=()):
    """Build objects and optional source hierarchy, preserving baked geometry."""
    used = set(existing_names)
    def unique(value):
        value = str(value)
        result, index = value, 1
        while result in used:
            result = f'{value} ({index})'
            index += 1
        used.add(result)
        return result
    objects = [Object(unique(name or Path(file_path).stem), mesh) for name, mesh in meshes]
    if not group or len(objects) < 2:
        return objects, objects[-1] if objects else None
    # Preserve named source parents when present. Their transforms are baked
    # into geometry by the loader; centered empty nodes retain the appearance.
    parents = {}
    for obj, (_, mesh) in zip(objects, meshes):
        path = tuple(mesh.source_hierarchy)
        for depth in range(1, len(path) + 1):
            key = path[:depth]
            if key not in parents:
                parents[key] = Object(unique(key[-1]), None)
        if path:
            obj.set_parent(parents[path])
    for path in sorted(parents, key=len, reverse=True):
        parent = parents[path]
        if parent.children:
            children = parent.children[:]
            bounds = [child.get_world_aabb() for child in children]
            center = (np.min([b[0] for b in bounds], axis=0) + np.max([b[1] for b in bounds], axis=0)) / 2
            for child in children:
                child.set_parent(None)
            parent.position = center.astype(np.float32)
            for child in children:
                child.set_parent(parent)
        if len(path) > 1:
            parent.set_parent(parents[path[:-1]])
    objects.extend(parents.values())
    roots = [obj for obj in objects if obj.parent is None]
    if len(roots) > 1:
        parent = group_objects(roots, unique(Path(file_path).stem))
        objects.append(parent)
    else:
        parent = roots[0]
    return objects, parent
