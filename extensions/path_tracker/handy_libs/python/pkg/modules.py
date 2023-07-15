import ast

def get_module_function_names(path):
    source = open(path).read()
    functions = [f.name for f in ast.parse(source).body
                if isinstance(f, ast.FunctionDef)]
    return functions