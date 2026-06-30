from typing import List

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class TableRow(PolygonBox):
    row_id: int

    @property
    def label(self) -> str:
        return f"Row {self.row_id}"


class TableCol(PolygonBox):
    col_id: int

    @property
    def label(self) -> str:
        return f"Column {self.col_id}"


class TableCell(PolygonBox):
    """Geometric cell derived from row × column intersection.

    The detector returns rows and columns only; cells are their geometric
    intersections, so no colspan/rowspan/header info is available."""

    row_id: int
    col_id: int
    cell_id: int

    @property
    def label(self) -> str:
        return f"Cell {self.cell_id}"


class TableResult(BaseModel):
    rows: List[TableRow]
    cols: List[TableCol]
    cells: List[TableCell]
    image_bbox: List[float]
    error: bool = False
