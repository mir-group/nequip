# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from nequip.data import AtomicDataDict


class NonPeriodicCellTransform(torch.nn.Module):
    """Construct a sensible orthogonal cell for nonperiodic systems without cells.

    For structures where PBC is all False and no cell is present (or override_cell
    is True), this transform constructs a large orthogonal cell that encompasses all
    atoms with appropriate padding. Positions are translated to ensure they fit within
    the constructed cell with positive coordinates.

    IMPORTANT: This transform MUST be applied BEFORE any neighborlist construction. Applying it after a neighborlist is constructed will invalidate the neighborlist and produce incorrect results.

    Args:
        padding: extra space to add around the structure in each direction (same units
                 as positions). Default is 10.0.
        override_cell: if True, replace existing cell even if one is present. Default is False.
    """

    def __init__(self, padding: float = 10.0, override_cell: bool = False):
        super().__init__()
        self.padding = padding
        self.override_cell = override_cell

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # TODO: add logic to detect and account for existing neighborlist (edge_index, edge_cell_shift, etc.) so that transform order doesn't matter.
        # For now, this MUST be called before neighborlist construction.

        # check if PBC exists and is all False; if not, this is a no-op
        if AtomicDataDict.PBC_KEY in data:
            pbc = data[AtomicDataDict.PBC_KEY]
            # (n_frames, 3) -> (n_frames,)
            is_non_periodic = ~pbc.any(dim=1)

            # no-op if any frame has any periodic boundary condition
            if not is_non_periodic.all():
                return data
        else:
            # no PBC key means we can't determine if nonperiodic, so no-op
            return data

        # check if cell is already present - if so, no-op unless override_cell is True
        if AtomicDataDict.CELL_KEY in data and not self.override_cell:
            return data

        # get positions
        pos = data[AtomicDataDict.POSITIONS_KEY]

        # determine number of frames
        n_frames = AtomicDataDict.num_frames(data)
        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY]
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)

        cells = []
        new_positions = pos.clone()

        for i in range(n_frames):
            frame_mask = batch == i
            frame_pos = pos[frame_mask]

            # find bounding box of the structure
            min_pos = torch.amin(frame_pos, dim=0)  # (3,)
            max_pos = torch.amax(frame_pos, dim=0)  # (3,)
            extent = max_pos - min_pos  # (3,)

            # translate so minimum coordinate is at padding
            new_positions[frame_mask] = frame_pos - min_pos + self.padding

            # construct orthogonal cell with padding on all sides
            cell = torch.diag(extent + 2 * self.padding).to(
                dtype=pos.dtype, device=pos.device
            )
            cells.append(cell)

        # update data with new cell and positions
        data[AtomicDataDict.CELL_KEY] = torch.stack(cells)
        data[AtomicDataDict.POSITIONS_KEY] = new_positions

        return data
