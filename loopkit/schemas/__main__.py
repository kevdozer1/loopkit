from .export import export_json_schemas

for path in export_json_schemas():
    print(path, flush=True)
