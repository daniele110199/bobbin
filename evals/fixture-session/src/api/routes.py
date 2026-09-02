"""HTTP routes for Depot."""

from src.api.auth import require_token
from src.inventory.stock import level_for, adjust
from src.inventory.reorder import reorder_due


@require_token
def get_stock(request):
    """GET /stock/<sku> — the current level for one item."""
    return {"sku": request.sku, "level": level_for(request.sku)}


@require_token
def post_adjust(request):
    """POST /stock/<sku>/adjust — move a level up or down by `delta`."""
    return {"sku": request.sku, "level": adjust(request.sku, request.delta)}


@require_token
def get_reorder(request):
    """GET /reorder — every sku currently below the reorder threshold."""
    return {"due": reorder_due()}
