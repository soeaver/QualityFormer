from lib.utils.registry import Registry


"""
Feature Extractor.
"""
# Backbone
BACKBONES = Registry()

# FPN
FPN_BODY = Registry()


"""
Instance Head.
"""

# Parsing Head
PARSING_HEADS = Registry()
PARSING_OUTPUTS = Registry()

PARSINGIOU_HEADS = Registry()
PARSINGIOU_OUTPUTS = Registry()

CDG_HEADS = Registry()

QEM_HEADS = Registry()
QEM_OUTPUTS = Registry()
