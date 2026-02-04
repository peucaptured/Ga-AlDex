
"""
Biome tile generator (Pokemon-like cohesion) using the user's Assets pack.

Updates vs previous version:
- Forces ALL ground/transition tiles to fill tile_px exactly (no transparent borders).
- Uses dark_grass_light_grass transition atlas to soften boundaries between dark and light grass.
- Removes paths entirely.
- Ensures water border tiles appear ONLY on shore; interior water uses water_core tiles.
- Stronger, smarter overlay usage:
  - Classifies forest overlays into tree vs shrub by analyzing sprite alpha bbox.
  - Places more overlays to make maps "alive" while keeping rules: only foam on water.
- Multi-tile overlay placement with occupancy grid (supports 2+ tile objects).

Dependencies: pillow, numpy
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import random
import json
import re
import math
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter


# ----------------------------
# Low-frequency noise utilities
# ----------------------------

def _blur2d(a: np.ndarray, iters: int = 2) -> np.ndarray:
    """Fast-ish blur using neighbor averaging."""
    a = a.astype(np.float32)
    for _ in range(iters):
        acc = (
            a
            + np.roll(a, 1, 0) + np.roll(a, -1, 0)
            + np.roll(a, 1, 1) + np.roll(a, -1, 1)
            + np.roll(np.roll(a, 1, 0), 1, 1) + np.roll(np.roll(a, 1, 0), -1, 1)
            + np.roll(np.roll(a, -1, 0), 1, 1) + np.roll(np.roll(a, -1, 0), -1, 1)
        )
        a = acc / 9.0
    return a


def value_noise(h: int, w: int, rng: random.Random, scale: int = 8, blur: int = 2) -> np.ndarray:
    """Creates smooth value noise in [0,1]."""
    gh = max(2, h // scale)
    gw = max(2, w // scale)
    grid = np.array([[rng.random() for _ in range(gw)] for _ in range(gh)], dtype=np.float32)
    up = np.kron(grid, np.ones((math.ceil(h / gh), math.ceil(w / gw)), dtype=np.float32))
    up = up[:h, :w]
    up = _blur2d(up, iters=blur)
    mn, mx = float(up.min()), float(up.max())
    if mx - mn < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    return (up - mn) / (mx - mn)


# ----------------------------
# Tile loading (mask tiles + atlas tiles)
# ----------------------------

_MASK_RE = re.compile(r"_m(\d{2})_v(\d+)\.png$", re.IGNORECASE)

@dataclass
class TileSet:
    name: str
    root: Path
    tile_raw_px: int  # not used for forced tiles, kept for reference
    plain: List[Path]
    masks: Dict[int, List[Path]]

    def pick_plain(self, rng: random.Random) -> Path:
        return rng.choice(self.plain)

    def pick_mask(self, mask: int, rng: random.Random) -> Optional[Path]:
        opts = self.masks.get(mask)
        if not opts:
            return None
        return rng.choice(opts)


def load_tileset(dir_path: Path, name: str, tile_raw_px: int) -> TileSet:
    plain: List[Path] = []
    masks: Dict[int, List[Path]] = {}
    for p in sorted(dir_path.glob("*.png")):
        if p.name.lower().endswith("_atlas.png"):
            continue
        if "ChatGPT Image" in p.name:
            # atlas sources
            continue
        m = _MASK_RE.search(p.name)
        if m:
            mask = int(m.group(1))
            masks.setdefault(mask, []).append(p)
        else:
            plain.append(p)
    return TileSet(name=name, root=dir_path, tile_raw_px=tile_raw_px, plain=plain, masks=masks)


@dataclass
class AtlasTile:
    id: str
    row: int
    col: int
    bbox: Tuple[int, int, int, int]  # x,y,w,h


class AtlasTileSet:
    """Tile set sourced from an atlas image + json with per-tile bbox."""
    def __init__(self, folder: Path, json_name: str):
        data = json.loads((folder / json_name).read_text(encoding="utf-8"))
        atlas_file = data.get("atlas_file") or data.get("atlas")
        if not atlas_file:
            raise ValueError(f"Atlas file not found in {json_name}")

        self.folder = folder
        self.atlas_path = folder / atlas_file
        if not self.atlas_path.exists():
            # some json uses just filename in folder root
            self.atlas_path = folder / Path(atlas_file).name

        self.atlas = Image.open(self.atlas_path).convert("RGBA")
        tiles = data.get("tiles") or []
        self.tiles: List[AtlasTile] = []
        for t in tiles:
            bb = t.get("atlas_bbox") or t.get("bbox_in_atlas") or {}
            self.tiles.append(AtlasTile(
                id=t.get("id",""),
                row=int(t.get("row",0)),
                col=int(t.get("col",0)),
                bbox=(int(bb.get("x",0)), int(bb.get("y",0)), int(bb.get("w",0)), int(bb.get("h",0)))
            ))

        # Sort stable row-major
        self.tiles.sort(key=lambda z: (z.row, z.col))
        self.by_id = {t.id: t for t in self.tiles}

    def crop(self, tile: AtlasTile) -> Image.Image:
        x, y, w, h = tile.bbox
        return self.atlas.crop((x, y, x + w, y + h))

    def pick_any(self, rng: random.Random) -> AtlasTile:
        return rng.choice(self.tiles)


# ----------------------------
# Overlay sprite loading + classification
# ----------------------------

@dataclass
class Sprite:
    path: Path
    w: int
    h: int
    pivot_x: int
    pivot_y: int
    tiles_w: int
    tiles_h: int


def load_sprite_entries(json_path: Path, base_dir: Path, tile_raw_px: int) -> List[Sprite]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    entries = data.get("entries") or []
    out: List[Sprite] = []
    for e in entries:
        file_rel = e.get("file")
        if not file_rel:
            continue
        p = base_dir / file_rel
        if not p.exists():
            p = base_dir / Path(file_rel).name
        w = int(e.get("w", 0))
        h = int(e.get("h", 0))
        piv = e.get("pivot") or {}
        pivot_x = int(piv.get("x", max(1, w // 2)))
        pivot_y = int(piv.get("y", max(1, h)))
        tiles_w = max(1, int(round(w / tile_raw_px)))
        tiles_h = max(1, int(round(h / tile_raw_px)))
        out.append(Sprite(path=p, w=w, h=h, pivot_x=pivot_x, pivot_y=pivot_y, tiles_w=tiles_w, tiles_h=tiles_h))
    return out


def alpha_bbox(im: Image.Image) -> Tuple[int,int,int,int]:
    """bbox of non-transparent pixels; returns (x0,y0,x1,y1) or full if none."""
    a = np.array(im.split()[-1])
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return (0,0,im.size[0], im.size[1])
    return (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))


# ----------------------------
# Generator
# ----------------------------

DEFAULT_ASSETS_ROOT = Path("Assets") / "map"


class BiomeGenerator:
    """
    Generate coherent biome maps as RGBA images.
    Output tiles are ALWAYS tile_px x tile_px.
    """

    def __init__(self, assets_root: str | Path = DEFAULT_ASSETS_ROOT):
        self.root = Path(assets_root)

        # Use water_grass tile size as raw reference (these are consistent)
        sample = next(iter((self.root / "water_grass").glob("water_grass_m00_v0.png")))
        self.tile_raw_px = Image.open(sample).size[0]

        # Atlas-based ground sets
        self.grass_atlas = AtlasTileSet(self.root / "grass", "grass.json")
        self.dark_grass_atlas = AtlasTileSet(self.root / "dark_grass", "dark_grass.json")
        self.light_dirt_atlas = AtlasTileSet(self.root / "light_dirt", "light_dirt.json")
        self.sand_atlas = AtlasTileSet(self.root / "sand", "sand.json")
        self.dark_light_trans_atlas = AtlasTileSet(self.root / "dark_grass_light_grass", "dark_grass_light_grass.json")
        self.wet_sand_tiles, self.dry_sand_tiles = self._load_wetdry_sand()

        # Mask-based transitions/terrains
        self.water_grass = load_tileset(self.root / "water_grass", "water_grass", self.tile_raw_px)
        self.water_sand  = load_tileset(self.root / "water_sand", "water_sand", self.tile_raw_px)
        self.water_core  = load_tileset(self.root / "water_core_tiles", "water_core", self.tile_raw_px)
        # Rock floor tiles (for cave base)
        self.rock_floor = load_tileset(self.root / "rocks", "rock_floor", self.tile_raw_px)
        self.river_narrow = load_tileset(self.root / "river_narrow_tiles", "river_narrow", self.tile_raw_px)
        self.dirt_rock_edge = load_tileset(self.root / "dirt_rock_edge", "dirt_rock_edge", self.tile_raw_px)

        # Water overlay tiles (foam etc.)
        self.water_overlay = load_tileset(self.root / "water_overlay", "water_overlay", self.tile_raw_px)

        # Overlays / objects
        self.rocks_sprites = load_sprite_entries(self.root / "rocks" / "rocks.json", self.root / "rocks", self.tile_raw_px)
        self.forest_sprites_all = load_sprite_entries(self.root / "forest_overlays" / "forest_overlays.json", self.root / "forest_overlays", self.tile_raw_px)
        self.foliage_sprites = load_sprite_entries(self.root / "foliage" / "foliage.json", self.root / "foliage", self.tile_raw_px)
        self.flower_sprites  = load_sprite_entries(self.root / "flower" / "flower.json", self.root / "flower", self.tile_raw_px)
        self.misc_sprites    = load_sprite_entries(self.root / "overlays_and_objects" / "asset.json", self.root / "overlays_and_objects", self.tile_raw_px)

        # Cache for resized images
        self._img_cache: Dict[Tuple[str, int, bool], Image.Image] = {}

        # Classify forest sprites into trees vs shrubs by analyzing alpha bbox aspect
        self.forest_trees, self.forest_shrubs = self._classify_forest_sprites()

        # Forest tree size control:
        # - Trees render-clamped to their footprint in _place_sprites (prevents giant canopies)
        # - Dense placement uses footprint=1x1 for most trees
        # - At most ONE 2x2 tree per map (max 4 squares)

        # Build a "big tree" candidate list from tree-like sprites that are visually larger in raw pixels.
        big_candidates: List[Sprite] = []
        for sp in self.forest_trees:
            try:
                im = Image.open(sp.path).convert("RGBA")
                x0,y0,x1,y1 = alpha_bbox(im)
                bw, bh = (x1-x0), (y1-y0)
                if bw >= int(self.tile_raw_px * 1.15) or bh >= int(self.tile_raw_px * 1.30):
                    big_candidates.append(sp)
            except Exception:
                continue
        if not big_candidates:
            big_candidates = list(self.forest_trees)

        self.forest_trees_big_2x2 = [self._clone_with_footprint(sp, 2, 2) for sp in big_candidates]

        # Dense forest pool: clone ALL trees to 1x1 footprint
        self.forest_trees_dense_1x1 = [self._clone_with_footprint(sp, 1, 1) for sp in self.forest_trees]


    # ---------- image IO ----------
    def _load_resized(self, p: Path, tile_px: int, force_tile: bool = False) -> Image.Image:
        """
        force_tile=True makes the image fill exactly (tile_px, tile_px).
        Use for ground/transition tiles.
        """
        key = (str(p), tile_px, force_tile)
        im = self._img_cache.get(key)
        if im is not None:
            return im
        src = Image.open(p).convert("RGBA")
        if force_tile:
            im = src.resize((tile_px, tile_px), resample=Image.NEAREST)
        else:
            scale = tile_px / float(self.tile_raw_px)
            w = max(1, int(round(src.size[0] * scale)))
            h = max(1, int(round(src.size[1] * scale)))
            im = src.resize((w, h), resample=Image.NEAREST)
        self._img_cache[key] = im
        return im

    def _load_sprite_with_shadow(self, p: Path, tile_px: int) -> Image.Image:
        key = (str(p), tile_px, "shadow")
        im = self._img_cache.get(key)
        if im is not None:
            return im
        base = self._load_resized(p, tile_px, force_tile=False)
        im = self._add_sprite_shadow(base)
        self._img_cache[key] = im
        return im

    @staticmethod
    def _add_sprite_shadow(im: Image.Image, offset: Tuple[int, int] = (0, 1)) -> Image.Image:
        alpha = im.split()[-1]
        shadow_mask = alpha.filter(ImageFilter.GaussianBlur(radius=1))
        shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
        shadow_layer = Image.new("RGBA", im.size, (0, 0, 0, 150))
        shadow.paste(shadow_layer, offset, shadow_mask)
        return Image.alpha_composite(shadow, im)

    def _crop_atlas_tile(self, atlas: AtlasTileSet, tile: AtlasTile, tile_px: int) -> Image.Image:
        """
        Crop from atlas and force to tile_px x tile_px (prevents border artifacts).
        """
        key = (f"{atlas.atlas_path}::{tile.id}", tile_px, True)
        im = self._img_cache.get(key)
        if im is not None:
            return im
        crop = atlas.crop(tile).convert("RGBA")
        im = crop.resize((tile_px, tile_px), resample=Image.NEAREST)
        self._img_cache[key] = im
        return im

    # ---------- masks ----------
    @staticmethod
    def _mask4(grid: np.ndarray, y: int, x: int, fn: Callable[[int], bool]) -> int:
        """bit=1 when neighbor satisfies fn"""
        h, w = grid.shape
        m = 0
        if y > 0     and fn(int(grid[y-1, x])): m |= 1
        if x < w-1   and fn(int(grid[y, x+1])): m |= 2
        if y < h-1   and fn(int(grid[y+1, x])): m |= 4
        if x > 0     and fn(int(grid[y, x-1])): m |= 8
        return m

    def _load_wetdry_sand(self) -> Tuple[List[Path], List[Path]]:
        wetdry_root = self.root / "wetdry_sand"
        wet = sorted(wetdry_root.glob("wetdry_sand_wet_*.png"))
        dry = sorted(wetdry_root.glob("wetdry_sand_dry_*.png"))
        if not wet or not dry:
            return [], []
        return wet, dry

    # ---------- distance ----------
    @staticmethod
    def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
        H, W = mask.shape
        dist = np.full((H, W), 1e9, dtype=np.float32)
        dist[mask] = 0.0
        for _ in range(3):
            for y in range(H):
                for x in range(W):
                    d = dist[y, x]
                    if y > 0: d = min(d, dist[y-1, x] + 1)
                    if x > 0: d = min(d, dist[y, x-1] + 1)
                    dist[y, x] = d
            for y in range(H-1, -1, -1):
                for x in range(W-1, -1, -1):
                    d = dist[y, x]
                    if y < H-1: d = min(d, dist[y+1, x] + 1)
                    if x < W-1: d = min(d, dist[y, x+1] + 1)
                    dist[y, x] = d
        return dist

    # ---------- forest sprite classification ----------
    def _is_tree_like(self, sp: Sprite) -> bool:
        """Heuristic tree detector (works for anonymous sprite filenames)."""
        try:
            im = Image.open(sp.path).convert("RGBA")
        except Exception:
            return False

        x0, y0, x1, y1 = alpha_bbox(im)
        bw, bh = (x1 - x0), (y1 - y0)
        if bw <= 1 or bh <= 1:
            return False

        crop = im.crop((x0, y0, x1, y1))
        arr = np.asarray(crop)
        a = arr[:, :, 3]
        mask = a > 20
        total = int(mask.sum())
        if total < 120:
            return False

        h, w = mask.shape
        # Bottom region should contain some brown-ish trunk pixels
        yb0 = int(h * 0.60)
        bottom = arr[yb0:, :, :3]
        bottom_a = (a[yb0:, :] > 20)

        if int(bottom_a.sum()) < 30:
            # If sprite is tall enough, accept as tree anyway
            return bh >= int(self.tile_raw_px * 1.15) and bw <= int(self.tile_raw_px * 1.80)

        R = bottom[:, :, 0].astype(np.int16)
        G = bottom[:, :, 1].astype(np.int16)
        B = bottom[:, :, 2].astype(np.int16)

        brown = (R > 85) & (G > 45) & (B < 95) & (R >= G) & bottom_a
        brown_count = int(brown.sum())
        brown_ratio = brown_count / float(max(1, int(bottom_a.sum())))

        # Brown pixels concentrated near center columns
        cx0 = int(w * 0.35)
        cx1 = int(w * 0.65)
        brown_center = int(brown[:, cx0:cx1].sum())
        brown_center_ratio = brown_center / float(max(1, brown_count))

        if (brown_ratio > 0.06 and brown_center_ratio > 0.25):
            return True

        # Fallback: tall-ish & not too wide
        return bh >= int(self.tile_raw_px * 1.20) and bw <= int(self.tile_raw_px * 1.70)

    def _classify_forest_sprites(self) -> Tuple[List[Sprite], List[Sprite]]:
        """
        Returns (tree_like, shrub_like) pools used by the FOREST biome.
        Uses both forest_overlays and foliage packs and detects trees by
        looking for trunk-like brown pixels near the bottom of the sprite.
        """
        trees: List[Sprite] = []
        shrubs: List[Sprite] = []

        combined = list(self.forest_sprites_all) + list(self.foliage_sprites)
        for sp in combined:
            if self._is_tree_like(sp):
                trees.append(sp)
            else:
                shrubs.append(sp)

        # Fallbacks
        if not trees:
            trees = list(self.forest_sprites_all)
        if not shrubs:
            shrubs = list(self.forest_sprites_all)

        return trees, shrubs


    @staticmethod
    def _split_exact_footprint(sprites: List[Sprite], tw: int, th: int) -> List[Sprite]:
        return [sp for sp in sprites if int(sp.tiles_w) == tw and int(sp.tiles_h) == th]

    @staticmethod
    def _split_small_1x1(sprites: List[Sprite]) -> List[Sprite]:
        return [sp for sp in sprites if int(sp.tiles_w) == 1 and int(sp.tiles_h) == 1]

    @staticmethod
    def _clone_with_footprint(sp: Sprite, tw: int, th: int) -> Sprite:
        return Sprite(
            path=sp.path,
            w=sp.w,
            h=sp.h,
            pivot_x=sp.pivot_x,
            pivot_y=sp.pivot_y,
            tiles_w=tw,
            tiles_h=th,
        )


    # ---------- overlay placement ----------
    def _place_sprites(
        self,
        canvas: Image.Image,
        rng: random.Random,
        sprites: List[Sprite],
        occ: np.ndarray,
        tile_px: int,
        attempts: int,
        density: float,
        allowed_anchor: np.ndarray,
        require_full_inside: bool = False,
    ) -> None:
        h, w = occ.shape
        target = int(h * w * density)
        placed = 0
        if not sprites:
            return

        for _ in range(attempts):
            if placed >= target:
                break
            sp = rng.choice(sprites)
            tw, th = sp.tiles_w, sp.tiles_h

            y = rng.randrange(th-1, h)
            x = rng.randrange(0, w - (tw-1))

            if not allowed_anchor[y, x]:
                continue

            y0 = y - (th-1)
            x0 = x
            y1 = y + 1
            x1 = x + tw
            if y0 < 0 or x1 > w:
                continue
            if occ[y0:y1, x0:x1].any():
                continue

            occ[y0:y1, x0:x1] = True

            im = self._load_sprite_with_shadow(sp.path, tile_px)

            # Ensure sprite visual size does not exceed its declared footprint (prevents giant trees/rocks)
            max_w_px = max(1, int(tw * tile_px))
            max_h_px = max(1, int(th * tile_px))
            if im.size[0] > max_w_px or im.size[1] > max_h_px:
                s = min(max_w_px / float(im.size[0]), max_h_px / float(im.size[1]))
                new_w = max(1, int(round(im.size[0] * s)))
                new_h = max(1, int(round(im.size[1] * s)))
                im = im.resize((new_w, new_h), resample=Image.NEAREST)
                extra_scale = s
            else:
                extra_scale = 1.0

            # Align pivot to bottom-center of footprint

            scale = (tile_px / float(self.tile_raw_px)) * float(extra_scale)
            tgt_x = int(round(x * tile_px + (tw * tile_px) / 2.0 - sp.pivot_x * scale))
            tgt_y = int(round((y + 1) * tile_px - sp.pivot_y * scale))
            # Skip if sprite would be completely outside canvas
            if tgt_x >= canvas.size[0] or tgt_y >= canvas.size[1] or (tgt_x + im.size[0]) <= 0 or (tgt_y + im.size[1]) <= 0:
                occ[y0:y1, x0:x1] = False
                continue

            # Optionally require the whole sprite to fit inside canvas (prevents clipping)
            if require_full_inside:
                if tgt_x < 0 or tgt_y < 0 or (tgt_x + im.size[0]) > canvas.size[0] or (tgt_y + im.size[1]) > canvas.size[1]:
                    occ[y0:y1, x0:x1] = False
                    continue

            canvas.alpha_composite(im, (tgt_x, tgt_y))
            placed += 1

    # ---------- biome grids ----------
    # Codes: 0=grass,1=dark_grass,2=light_dirt,3=sand,4=rock,5=water
    def _make_sea(self, H: int, W: int, rng: random.Random):
        n = value_noise(H, W, rng, scale=7, blur=3)
        water = n < 0.45

        # flood-fill ocean connected to border
        from collections import deque
        vis = np.zeros((H, W), dtype=bool)
        q = deque()
        for x in range(W):
            if water[0, x]: q.append((0, x)); vis[0, x] = True
            if water[H-1, x]: q.append((H-1, x)); vis[H-1, x] = True
        for y in range(H):
            if water[y, 0]: q.append((y, 0)); vis[y, 0] = True
            if water[y, W-1]: q.append((y, W-1)); vis[y, W-1] = True
        while q:
            y, x = q.popleft()
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W and water[ny, nx] and not vis[ny, nx]:
                    vis[ny, nx] = True
                    q.append((ny, nx))

        ocean = vis
        dist = self._distance_to_mask(~ocean)
        sand = (~ocean) & (dist <= 2.5)
        grass = (~ocean) & (~sand)
        return ocean, sand, grass

    def _make_river(self, H: int, W: int, rng: random.Random):
        y = rng.randrange(H//4, 3*H//4)
        x = 0
        path = np.zeros((H, W), dtype=bool)
        while x < W:
            path[y, x] = True
            x += 1
            if rng.random() < 0.55:
                y += rng.choice([-1, 0, 1])
                y = max(1, min(H-2, y))

        river = path.copy()
        for _ in range(2):
            river = river | np.roll(river, 1, 0) | np.roll(river, -1, 0)
        if rng.random() < 0.5:
            river = river | np.roll(river, 1, 1)

        land = ~river
        dist_to_river = self._distance_to_mask(river)
        sand = land & (dist_to_river <= 2.0)
        grass = land & (~sand)
        return river, sand, grass

    def _make_forest(self, H: int, W: int, rng: random.Random):
        n = value_noise(H, W, rng, scale=9, blur=3)
        dark = n < 0.72
        dark = _blur2d(dark.astype(np.float32), iters=1) > 0.45
        clear = ~dark
        return dark, clear

    def _make_prairie(self, H: int, W: int, rng: random.Random):
        n = value_noise(H, W, rng, scale=10, blur=3)
        dirt = n < 0.18
        dirt = _blur2d(dirt.astype(np.float32), iters=1) > 0.55
        grass = ~dirt
        return grass, dirt

    def _make_dirt(self, H: int, W: int, rng: random.Random):
        n = value_noise(H, W, rng, scale=9, blur=3)
        grass = n > 0.84
        dirt = ~grass
        return dirt, grass

    def _make_desert(self, H: int, W: int, rng: random.Random):
        sand = np.ones((H, W), dtype=bool)
        damp = value_noise(H, W, rng, scale=9, blur=3) < 0.22
        return sand, damp

    def _make_cave(self, H: int, W: int, rng: random.Random):
        n = value_noise(H, W, rng, scale=9, blur=3)
        rock = n < 0.62
        rock = _blur2d(rock.astype(np.float32), iters=1) > 0.5
        dirt = ~rock
        return rock, dirt

    
    # ---------- procedural grass<->sand transition (option 1) ----------
    def _edge_mask(self, tile_px: int, direction: str, rng: random.Random, band: int = 14) -> Image.Image:
        """
        Create an alpha mask (L) for blending an edge band from one tile into another.
        direction: 'N','E','S','W' indicating where the OTHER material is coming from.
        band: width in pixels of the transition band.
        """
        band = max(6, min(tile_px // 2, band))
        a = np.zeros((tile_px, tile_px), dtype=np.float32)

        # Base ramp
        if direction == 'N':
            for y in range(tile_px):
                t = 1.0 - min(1.0, y / float(band))
                a[y, :] = t
        elif direction == 'S':
            for y in range(tile_px):
                t = 1.0 - min(1.0, (tile_px - 1 - y) / float(band))
                a[y, :] = t
        elif direction == 'W':
            for x in range(tile_px):
                t = 1.0 - min(1.0, x / float(band))
                a[:, x] = t
        else:  # 'E'
            for x in range(tile_px):
                t = 1.0 - min(1.0, (tile_px - 1 - x) / float(band))
                a[:, x] = t

        # Add small noise so it doesn't look like a perfect stripe
        noise = np.array([[rng.random() for _ in range(tile_px)] for _ in range(tile_px)], dtype=np.float32)
        noise = _blur2d(noise, iters=1)
        a = np.clip(a * (0.85 + 0.35 * noise), 0.0, 1.0)

        # Slight threshold to avoid over-blending
        a = np.clip((a - 0.10) / 0.90, 0.0, 1.0)

        mask = Image.fromarray((a * 255).astype(np.uint8), mode="L")
        return mask

    def _blend_tiles(self, base: Image.Image, overlay: Image.Image, mask: Image.Image) -> Image.Image:
        """Return base with overlay blended using mask."""
        out = base.copy()
        out.paste(overlay, (0, 0), mask)
        return out

    def _lighten_tile(self, im: Image.Image, amount: float) -> Image.Image:
        enhancer = ImageEnhance.Brightness(im)
        return enhancer.enhance(amount)

    def _smooth_water_transition(
        self,
        base: Image.Image,
        tile_px: int,
        rng: random.Random,
        touches: List[str],
        amount: float = 1.18
    ) -> Image.Image:
        if not touches:
            return base
        overlay = self._lighten_tile(base, amount)
        out = base
        for d in touches:
            mask = self._edge_mask(tile_px, d, rng, band=max(8, tile_px // 6))
            out = self._blend_tiles(out, overlay, mask)
        return out

    def _apply_grass_sand_transition(self, grid: np.ndarray, x: int, y: int, tile_px: int, rng: random.Random, base_tile: Image.Image) -> Image.Image:
        """
        If current cell is sand (3) and touches grass (0/1), blend a thin grass band into sand along touching edges.
        This creates a Pokemon-like soft shore between grass and sand without needing dedicated tiles.
        """
        H, W = grid.shape
        if int(grid[y, x]) != 3:
            return base_tile

        # Determine which sides touch grass/dark_grass
        touches = []
        if y > 0 and int(grid[y-1, x]) in (0, 1): touches.append('N')
        if x < W-1 and int(grid[y, x+1]) in (0, 1): touches.append('E')
        if y < H-1 and int(grid[y+1, x]) in (0, 1): touches.append('S')
        if x > 0 and int(grid[y, x-1]) in (0, 1): touches.append('W')

        if not touches:
            return base_tile

        # Use light grass as overlay band (looks like fringe)
        grass_tile = self._crop_atlas_tile(self.grass_atlas, self.grass_atlas.pick_any(rng), tile_px)
        out = base_tile
        for d in touches:
            mask = self._edge_mask(tile_px, d, rng, band=max(10, tile_px // 5))
            out = self._blend_tiles(out, grass_tile, mask)
        return out

# ---------- public API ----------
    def generate(self, biome: str, grid_w: int = 32, grid_h: int = 32, tile_px: int = 64, seed: int = 0) -> Image.Image:
        rng = random.Random(seed)
        biome = biome.lower().strip()

        grid = np.zeros((grid_h, grid_w), dtype=np.int8)
        dist_to_land: Optional[np.ndarray] = None
        damp_mask: Optional[np.ndarray] = None

        if biome == "sea":
            ocean, sand, grass = self._make_sea(grid_h, grid_w, rng)
            grid[ocean] = 5
            grid[sand] = 3
            grid[grass] = 0
            dist_to_land = self._distance_to_mask(~(grid == 5))

        elif biome == "river":
            river, sand, grass = self._make_river(grid_h, grid_w, rng)
            grid[river] = 5
            grid[sand] = 3
            grid[grass] = 0
            dist_to_land = self._distance_to_mask(~(grid == 5))

        elif biome == "center_lake":
            grid[:, :] = 0
            cr, cc = grid_h // 2, grid_w // 2
            if grid_w <= 6 or grid_h <= 6:
                rad = 1
            elif grid_w >= 12 or grid_h >= 12:
                rad = 3
            else:
                rad = 2
            for y in range(grid_h):
                for x in range(grid_w):
                    if (y - cr) ** 2 + (x - cc) ** 2 <= (rad + 0.35) ** 2:
                        grid[y, x] = 5
            dist_to_land = self._distance_to_mask(~(grid == 5))

        elif biome == "forest":
            dark, clear = self._make_forest(grid_h, grid_w, rng)
            grid[dark] = 1
            grid[clear] = 0

        elif biome == "prairie":
            grass, dirt = self._make_prairie(grid_h, grid_w, rng)
            grid[grass] = 0
            grid[dirt] = 2

        elif biome == "dirt":
            dirt, grass = self._make_dirt(grid_h, grid_w, rng)
            grid[dirt] = 2
            grid[grass] = 0

        elif biome == "desert":
            sand, damp = self._make_desert(grid_h, grid_w, rng)
            grid[sand] = 3
            # Use damp mask to bias wet/dry sand selection in ground layer
            damp_mask = damp.astype(np.bool_)

        elif biome == "cave":
            rock, dirt = self._make_cave(grid_h, grid_w, rng)
            grid[rock] = 4
            grid[dirt] = 2

        else:
            raise ValueError(f"Unknown biome: {biome}")

        canvas = Image.new("RGBA", (grid_w * tile_px, grid_h * tile_px), (0,0,0,0))

        def blit_tile(x: int, y: int, im: Image.Image):
            canvas.alpha_composite(im, (x * tile_px, y * tile_px))

        # --- Ground layer (force-fill tiles) ---
        for y in range(grid_h):
            for x in range(grid_w):
                t = int(grid[y, x])
                if t == 1:
                    tile = self.dark_grass_atlas.pick_any(rng)
                    blit_tile(x, y, self._crop_atlas_tile(self.dark_grass_atlas, tile, tile_px))
                elif t == 2:
                    tile = self.light_dirt_atlas.pick_any(rng)
                    blit_tile(x, y, self._crop_atlas_tile(self.light_dirt_atlas, tile, tile_px))
                elif t == 3:
                    # Sand base with optional wet/dry variation (sea beaches + desert damp patches)
                    if self.wet_sand_tiles and self.dry_sand_tiles:
                        if biome == "sea" and dist_to_land is not None:
                            use_wet = float(dist_to_land[y, x]) <= 1.5
                            tilep = rng.choice(self.wet_sand_tiles if use_wet else self.dry_sand_tiles)
                            sand_im = self._load_resized(tilep, tile_px, force_tile=True)
                        elif biome == "desert" and damp_mask is not None:
                            use_wet = bool(damp_mask[y, x])
                            # soften: avoid large blotches looking too binary
                            if rng.random() < 0.15:
                                use_wet = not use_wet
                            tilep = rng.choice(self.wet_sand_tiles if use_wet else self.dry_sand_tiles)
                            sand_im = self._load_resized(tilep, tile_px, force_tile=True)
                        else:
                            tile = self.sand_atlas.pick_any(rng)
                            sand_im = self._crop_atlas_tile(self.sand_atlas, tile, tile_px)
                    else:
                        tile = self.sand_atlas.pick_any(rng)
                        sand_im = self._crop_atlas_tile(self.sand_atlas, tile, tile_px)

                    sand_im = self._apply_grass_sand_transition(grid, x, y, tile_px, rng, sand_im)
                    blit_tile(x, y, sand_im)
                elif t == 4:
                    # Cave rock base: prefer dedicated rock floor tiles; edges added below
                    if getattr(self, "rock_floor", None) is not None and getattr(self.rock_floor, "plain", None):
                        tilep = self.rock_floor.pick_plain(rng)
                        blit_tile(x, y, self._load_resized(tilep, tile_px, force_tile=True))
                    else:
                        tile = self.light_dirt_atlas.pick_any(rng)
                        blit_tile(x, y, self._crop_atlas_tile(self.light_dirt_atlas, tile, tile_px))
                else:
                    tile = self.grass_atlas.pick_any(rng)
                    blit_tile(x, y, self._crop_atlas_tile(self.grass_atlas, tile, tile_px))

# --- Transitions / special terrains ---
        for y in range(grid_h):
            for x in range(grid_w):
                t = int(grid[y, x])

                # Water: interior uses water_core; shores use water_grass/water_sand
                # Water: interior uses water_core (shallow ring + deep core); shores use water_grass/water_sand
                if t == 5:
                    def neigh_is_land(v: int) -> bool:
                        return v != 5
                
                    land_mask = self._mask4(grid, y, x, neigh_is_land)
                
                    # --- helpers robustos (não dependem de atributos existirem) ---
                    def _pick_from_list(lst, rng):
                        return rng.choice(lst) if lst else None
                                    
                    def _norm_str(p) -> str:
                        if p is None:
                            return ""
                        # se vier Path, tuple, TileRef, etc., vira string
                        try:
                            return str(p).lower()
                        except Exception:
                            return ""
                    
                    def _is_deep_path(p) -> bool:
                        s = _norm_str(p)
                        return ("deep" in s) or ("_d" in s and "shallow" not in s)
                    
                    def _is_shallow_path(p) -> bool:
                        s = _norm_str(p)
                        return ("shallow" in s) or ("raso" in s)
                
                    def _pick_water_core_prefer(prefer: str, rng) -> str:
                        """
                        prefer: 'shallow' | 'deep' | 'any'
                        Tenta usar listas especializadas se existirem; senão tenta várias vezes pick_plain.
                        """
                        # Se você tiver listas separadas no objeto, usa
                        shallow_lst = getattr(self, "water_shallow_tiles", None) or getattr(self, "water_core_shallow_tiles", None)
                        deep_lst    = getattr(self, "water_deep_tiles", None)    or getattr(self, "water_core_deep_tiles", None)
                
                        if prefer == "shallow" and shallow_lst:
                            return _pick_from_list(shallow_lst, rng)
                        if prefer == "deep" and deep_lst:
                            return _pick_from_list(deep_lst, rng)
                
                        # Se o seu water_core tiver listas internas, tenta descobrir
                        cand = [p for p in plain_lst if isinstance(p, (str,)) or hasattr(p, "__fspath__")]
                        if isinstance(plain_lst, (list, tuple)) and plain_lst:
                            if prefer == "shallow":
                                cand = [p for p in plain_lst if _is_shallow_path(p)]
                                if cand:
                                    return _pick_from_list(cand, rng)
                            if prefer == "deep":
                                cand = [p for p in plain_lst if _is_deep_path(p)]
                                if cand:
                                    return _pick_from_list(cand, rng)
                
                        # Fallback: tenta pick_plain algumas vezes até bater o "prefer"
                        if prefer in ("shallow", "deep"):
                            for _ in range(12):
                                p = self.water_core.pick_plain(rng)
                                if p is None:
                                    break
                                if prefer == "shallow" and _is_shallow_path(p):
                                    return p
                                if prefer == "deep" and _is_deep_path(p):
                                    return p
                
                        # Ultimíssimo fallback
                        return self.water_core.pick_plain(rng)
                
                    if land_mask == 0:
                        # interior water only (sem terra colada)
                        # >>> CORREÇÃO PRINCIPAL: shallow perto da margem, deep no miolo <<<
                        prefer = "any"
                        if dist_to_land is not None:
                            d = float(dist_to_land[y, x])
                            if d <= 1.5:
                                prefer = "shallow"
                            elif d >= 3.5:
                                prefer = "deep"
                            else:
                                # faixa de mistura: tende ao shallow, mas pode variar
                                prefer = "shallow" if rng.random() < 0.70 else "deep"
                
                        tilep = _pick_water_core_prefer(prefer, rng)
                        if tilep is None:
                            tilep = self.water_core.pick_plain(rng)
                
                        water_im = self._load_resized(tilep, tile_px, force_tile=True)
                
                        # mantém seu smoothing (agora ele trabalha em cima de um core mais coerente)
                        if dist_to_land is not None and dist_to_land[y, x] <= 2.5:
                            touches = []
                            for dy, dx, dch in [(-1,0,'N'),(1,0,'S'),(0,-1,'W'),(0,1,'E')]:
                                ny, nx = y+dy, x+dx
                                if 0 <= ny < grid_h and 0 <= nx < grid_w and dist_to_land[ny, nx] <= 1.5:
                                    touches.append(dch)
                            water_im = self._smooth_water_transition(water_im, tile_px, rng, touches)
                
                        blit_tile(x, y, water_im)
                
                    else:
                        # shoreline tiles depending on adjacent sand
                        adj_has_sand = False
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < grid_h and 0 <= nx < grid_w and int(grid[ny, nx]) == 3:
                                adj_has_sand = True
                                break
                
                        tilep = (self.water_sand.pick_mask(land_mask, rng) if adj_has_sand
                                 else self.water_grass.pick_mask(land_mask, rng))
                
                        if tilep is None:
                            # fallback: se faltou tile de borda por algum motivo, usa shallow (não deep) perto de margem
                            tilep = _pick_water_core_prefer("shallow", rng) or self.water_core.pick_plain(rng)
                
                        blit_tile(x, y, self._load_resized(tilep, tile_px, force_tile=True))


        # --- Overlay layer ---
        occ = np.zeros((grid_h, grid_w), dtype=bool)

        is_water = grid == 5
        is_land = ~is_water
        is_grass = (grid == 0) | (grid == 1)

        # Rule: only foam tiles on water, and only near shore
        shore_water = np.zeros_like(is_water)
        for y in range(grid_h):
            for x in range(grid_w):
                if not is_water[y, x]:
                    continue
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < grid_h and 0 <= nx < grid_w and is_land[ny, nx]:
                        shore_water[y, x] = True
                        break

        if biome in ("sea", "river") and dist_to_land is not None:
            for y in range(grid_h):
                for x in range(grid_w):
                    if dist_to_land[y, x] == 2 and rng.random() < (0.30 if biome == "sea" else 0.18):
                        if self.water_overlay.plain:
                            tilep = self.water_overlay.pick_plain(rng)
                            blit_tile(x, y, self._load_resized(tilep, tile_px, force_tile=True))

        # Rocks: in all biomes, on land only
        rock_density = 0.0 if biome == "forest" else (0.030 if biome == "sea" else 0.020)
        self._place_sprites(canvas, rng, self.rocks_sprites, occ, tile_px, attempts=1600, density=rock_density, allowed_anchor=is_land)

        # Forest: dense 1x1 trees + shrubs, and at most ONE 2x2 "big tree"
        if biome == "forest":
            # 1) Place at most one big 2x2 tree (max 4 squares)
            if self.forest_trees_big_2x2:
                self._place_sprites(
                    canvas, rng,
                    self.forest_trees_big_2x2,
                    occ, tile_px,
                    attempts=2000,
                    density=(1.0 / max(1, (grid_h * grid_w))),  # target ~= 1
                    allowed_anchor=is_grass,
                    require_full_inside=True,
                )

            # 2) Dense small trees (prefer 1x1)
            self._place_sprites(
                canvas, rng,
                self.forest_trees_dense_1x1,
                occ, tile_px,
                attempts=20000,
                density=0.35,
                allowed_anchor=is_grass,
                require_full_inside=True,
            )

            # 3) Shrubs / undergrowth
            self._place_sprites(
                canvas, rng,
                self.forest_shrubs,
                occ, tile_px,
                attempts=9000,
                density=0.06,
                allowed_anchor=is_grass,
                require_full_inside=True,
            )

            # 4) Extra foliage (usually small plants)
            self._place_sprites(
                canvas, rng,
                self.foliage_sprites,
                occ, tile_px,
                attempts=8000,
                density=0.05,
                allowed_anchor=is_grass,
                require_full_inside=True,
            )

        # Prairie: flowers + shrubs, few trees (only if they are small)
        if biome == "prairie":
            self._place_sprites(canvas, rng, self.flower_sprites, occ, tile_px, attempts=4500, density=0.08, allowed_anchor=is_grass)
            self._place_sprites(canvas, rng, self.foliage_sprites, occ, tile_px, attempts=3500, density=0.05, allowed_anchor=is_grass)
            # tiny chance of a shrub from forest pack
            self._place_sprites(canvas, rng, self.forest_shrubs, occ, tile_px, attempts=1200, density=0.01, allowed_anchor=is_grass)

        # Dirt: some sparse foliage
        if biome == "dirt":
            self._place_sprites(canvas, rng, self.foliage_sprites, occ, tile_px, attempts=2000, density=0.02, allowed_anchor=is_grass)

        # Desert: sparse misc (land only)
        if biome == "desert":
            self._place_sprites(canvas, rng, self.misc_sprites, occ, tile_px, attempts=1600, density=0.015, allowed_anchor=is_land)

        # Cave: sparse misc (land only) + rocks already
        if biome == "cave":
            self._place_sprites(canvas, rng, self.misc_sprites, occ, tile_px, attempts=2500, density=0.03, allowed_anchor=is_land)

        return canvas


def generate_to_file(
    out_png: str | Path,
    biome: str,
    assets_root: str | Path = DEFAULT_ASSETS_ROOT,
    grid_w: int = 32,
    grid_h: int = 32,
    tile_px: int = 64,
    seed: int = 0,
) -> None:
    gen = BiomeGenerator(assets_root)
    img = gen.generate(biome=biome, grid_w=grid_w, grid_h=grid_h, tile_px=tile_px, seed=seed)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
