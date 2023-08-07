
# QuTreeNano

QuTreeNano is a basic Tensor Train simulator focused on simplicity and easy of use.
When the original QuTree package is an excavator, QuTreeNano is a shovel: not the
best tool for big tasks but small, easy to use, easy to take and use immediately.

Here are the central simplification principles:
- TrensorTrains (TTs, aka as Matrix-Product States (MPS)) are the only tree structure
- Leaf-dimensions are powers of 2 (2**n). Doesn't fit? Pad it.
- same rank at every edge (or *bond*)
- code & usage simplicity over performance


