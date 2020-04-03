"""Misc helper functions
"""


def open_text_file(filepath: str, mode: str = 'r'):
    return open(filepath, mode=mode, encoding='utf-8', newline='\n')
