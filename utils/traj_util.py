def validate_trajectory_tree(tree, *, _level=0, _path="root"):
    """
    Recursively validate that `tree` is a proper trajectory-tree dictionary.

    Rules
    -----
    1. The outermost object must be a `dict` with one and only one key: 'root'.
    2. Every key at every level must be a `str`.
    3. Every value must be a `dict`.
    4. Leaves must be *empty* dicts: {}.
    5. No other data types (lists, tuples, sets, numbers, None, ...) are allowed.

    Parameters
    ----------
    tree : Any
        The candidate trajectory tree.
    _level, _path : int, str
        Internal recursion helpers (do not use).

    Returns
    -------
    True
        If the tree is valid.

    Raises
    ------
    ValueError
        If a rule is violated, with an explanation of the first problem found.
    """
    # ── Rule 1 : Only the top level is allowed to have a single 'root' key ──
    if _level == 0:
        if not isinstance(tree, dict):
            return False
        if list(tree.keys()) != ["root"]:
            return False
        # descend to its value
        return validate_trajectory_tree(tree["root"], _level=1, _path="root")

    # From here on we are inside the subtree rooted at `root`
    if not isinstance(tree, dict):
        return False

    for key, sub in tree.items():
        if not isinstance(key, str):
            return False
        # If value is empty dict → leaf (Rule 4)
        if sub == {}:
            continue
        # Otherwise recurse, enforcing Rule 3
        if not isinstance(sub, dict):
            return False
        validate_trajectory_tree(sub, _level=_level + 1, _path=f"{_path}->{key}")

    return True

