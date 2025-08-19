import re

def convert_formula(expr:str, var_map:dict=None):
    """
    Convert a symbolic formula (gplearn style) to a valid expression.
    Supported functions: add, sub, mul, div, sqrt, cos
    Example: 'add(div(0.097, X0),X1)' -> '(0.097 / x1 + x2)'
    """
    if var_map is None:
        var_map = {'X0': 'x1', 'X1': 'x2', 'X2': 'x3'}  # Customize as needed

    func_map = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
        'sqrt': 'np.sqrt',
        'cos': 'np.cos',
    }

    def replace_vars(s):
        for k, v in var_map.items():
            s = s.replace(k, v)
        return s

    def parse(expr):
        # Base case: if no function, just replace variables
        if not any(f in expr for f in func_map):
            return replace_vars(expr)
        
        # Recursive case: parse function calls
        for func, op in func_map.items():
            pattern = rf'{func}\(([^(),]+(?:\([^()]*\)[^(),]*)*),([^(),]+(?:\([^()]*\)[^(),]*)*)\)'
            while re.search(pattern, expr):
                def repl(m):
                    a = parse(m.group(1).strip())
                    b = parse(m.group(2).strip())
                    if func in ['add', 'sub', 'mul', 'div']:
                        return f'({a} {op} {b})'
                    else:
                        return f'{op}({a})'
                expr = re.sub(pattern, repl, expr)
        return expr

    return parse(expr)