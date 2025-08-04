# Here we have the globally defined configures for the entire credmodex module
# This allows the user more flexibility when using the arquitechture and improves adaptability


__all__ = [
    'get_forbidden_cols',
    'DEFAULT_FORBIDDEN_COLS',
    'add_forbidden_cols',
    'set_column_alias',
]


# A dictionary to allow the user to override default column names
# can override with {'date': 'DATA'} for example
DEFAULT_FORBIDDEN_COLS = {
    'id': 'id', 
    'target': 'target', 
    'date': 'date',
    'split': 'split', 
    'score': 'score', 
    'rating': 'rating', 
    'loan_amount': 'loan_amount', 
    'term': 'term', 
}


# Optional utility functions
def get_forbidden_cols(additional_cols:list=None):
    """
    Returns the global forbidden columns list, applying COLUMN_ALIASES.
    Optionally adds more user-defined forbidden columns.
    """
    cols = list(set(DEFAULT_FORBIDDEN_COLS.values()))
    if additional_cols:
        cols += additional_cols
    return list(set(cols))


def set_column_alias(original_name: str, new_name: str):
    """
    Override default column names like 'date' → 'DATA'
    """
    DEFAULT_FORBIDDEN_COLS[original_name] = new_name


def add_forbidden_cols(additional_cols:list=None):
    for col in additional_cols:
        DEFAULT_FORBIDDEN_COLS[col] = col
    return





if __name__ == '__main__':
    DEFAULT_FORBIDDEN_COLS['id'] = 'id_'
    print(add_forbidden_cols(additional_cols=['loss', 'OIIIEEE']))
    print(DEFAULT_FORBIDDEN_COLS)