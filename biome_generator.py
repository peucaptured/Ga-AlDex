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
- Cave biome SPECIFIC update:
  - Generates a clean ARENA (rectangular floor surrounded by walls).
  - Uses 'sand' assets for the floor (interior).
  - Uses 'dirt_rock_edge' for walls, with brown/dirt edge facing the sand interior.
  - ONLY uses 'cave_overlay' assets for decoration, with reduced quantity.

Dependencies: pillow, numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
import random
import json
import re
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


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
    tile_raw_px: int
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
    def __init__(self, folder: Path, json_name: str):
        data = json.loads((folder / json_name).read_text(encoding="utf-8"))
        atlas_file = data.get("atlas_file") or data.get("atlas")
        if not atlas_file:
            raise ValueError(f"Atlas file not found in {json_name}")

        self.folder = folder
        self.atlas_path = folder / atlas_file
        if not self.atlas_path.exists():
            self.atlas_path = folder / Path(atlas_file).name

        self.atlas = Image.open(self.atlas_path).convert("RGBA")
        tiles = data.get("tiles") or []
        self.tiles: List[AtlasTile] = []
        for t in tiles:
            bb = t.get("atlas_bbox") or t.get("bbox_in_atlas") or {}
            self.tiles.append(
                AtlasTile(
                    id=t.get("id", ""),
                    row=int(t.get("row", 0)),
                    col=int(t.get("col", 0)),
                    bbox=(int(bb.get("x", 0)), int(bb.get("y", 0)), int(bb.get("w", 0)), int(bb.get("h", 0))),
                )
            )
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
    if not json_path.exists():
        return []
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


def alpha_bbox(im: Image.Image) -> Tuple[int, int, int, int]:
    a = np.array(im.split()[-1])
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return (0, 0, im.size[0], im.size[1])
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))


# ----------------------------
# Generator
# ----------------------------

DEFAULT_ASSETS_ROOT = Path("Assets") / "map"


class BiomeGenerator:
    """
    Generate coherent biome maps as RGBA images.
    Output tiles are ALWAYS tile_px x tile_px.
    """

    def apply_cave_lighting(self, canvas: Image.Image) -> Image.Image:
        # [REMOVIDO] - Sem iluminação escura
        return canvas

    def __init__(self, assets_root: str | Path = DEFAULT_ASSETS_ROOT):
        self.root = Path(assets_root)

        # Use water_grass tile size as raw reference (these are consistent)
        sample_iter = (self.root / "water_grass").glob("water_grass_m00_v0.png")
        sample = next(sample_iter, None)
        if sample:
            self.tile_raw_px = Image.open(sample).size[0]
        else:
            self.tile_raw_px = 64 # Fallback

        # Atlas-based ground sets
        self.grass_atlas = AtlasTileSet(self.root / "grass", "grass.json")
        self.dark_grass_atlas = AtlasTileSet(self.root / "dark_grass", "dark_grass.json")
        self.light_dirt_atlas = AtlasTileSet(self.root / "light_dirt", "light_dirt.json")
        # NOTE: some code paths refer to this as `sand_atlas`.
        # Keep a stable attribute name to avoid AttributeError.
        self.sand_atlas = AtlasTileSet(self.root / "sand", "sand.json")
        self.sand = self.sand_atlas
        self.dark_grass_light_grass = AtlasTileSet(self.root / "dark_grass_light_grass", "dark_grass_light_grass.json")
        self.wet_sand_tiles, self.dry_sand_tiles = self._load_wetdry_sand()

        # Mask-based transitions/terrains
        self.water_grass = load_tileset(self.root / "water_grass", "water_grass", self.tile_raw_px)
        self.water_sand = load_tileset(self.root / "water_sand", "water_sand", self.tile_raw_px)
        self.water_core = load_tileset(self.root / "water_core_tiles", "water_core", self.tile_raw_px)
        # Rock floor tiles (for cave base)
        self.rock_floor = load_tileset(self.root / "rocks", "rock_floor", self.tile_raw_px)
        self.river_narrow = load_tileset(self.root / "river_narrow_tiles", "river_narrow", self.tile_raw_px)
        self.dirt_rock_edge = load_tileset(self.root / "dirt_rock_edge", "dirt_rock_edge", self.tile_raw_px)

        # Water overlay tiles (foam etc.)
        self.water_overlay = load_tileset(self.root / "water_overlay", "water_overlay", self.tile_raw_px)

        # Overlays / objects
        self.rocks_sprites = load_sprite_entries(self.root / "rocks" / "rocks.json", self.root / "rocks", self.tile_raw_px)
        self.forest_sprites_all = load_sprite_entries(
            self.root / "forest_overlays" / "forest_overlays.json", self.root / "forest_overlays", self.tile_raw_px
        )
        self.foliage_sprites = load_sprite_entries(self.root / "foliage" / "foliage.json", self.root / "foliage", self.tile_raw_px)
        self.flower_sprites = load_sprite_entries(self.root / "flower" / "flower.json", self.root / "flower", self.tile_raw_px)
        self.misc_sprites = load_sprite_entries(
            self.root / "overlays_and_objects" / "asset.json", self.root / "overlays_and_objects", self.tile_raw_px
        )
        # Assets específicos da caverna
        self.cave_sprites = load_sprite_entries(
            self.root / "cave_overlay" / "cave_overlay.json", self.root / "cave_overlay", self.tile_raw_px
        )

        # Cache for resized images
        self._img_cache: Dict[Tuple[str, int, Any], Optional[Image.Image]] = {}

        # Classify forest sprites into trees vs shrubs by analyzing alpha bbox aspect
        self.forest_trees, self.forest_shrubs = self._classify_forest_sprites()

    # ---------- image IO ----------
    def _load_resized(self, p: Path, tile_px: int, force_tile: bool = False) -> Optional[Image.Image]:
        key = (str(p), tile_px, force_tile)
        if key in self._img_cache:
            return self._img_cache[key]

        try:
            src = Image.open(p).convert("RGBA")
        except FileNotFoundError:
            self._img_cache[key] = None
            return None

        if force_tile:
            im = src.resize((tile_px, tile_px), resample=Image.NEAREST)
        else:
            scale = tile_px / float(self.tile_raw_px)
            w = max(1, int(round(src.size[0] * scale)))
            h = max(1, int(round(src.size[1] * scale)))
            im = src.resize((w, h), resample=Image.NEAREST)

        self._img_cache[key] = im
        return im

    def _load_sprite_with_shadow(self, p: Path, tile_px: int) -> Optional[Image.Image]:
        key = (str(p), tile_px, "shadow")
        if key in self._img_cache:
            return self._img_cache[key]

        base = self._load_resized(p, tile_px, force_tile=False)
        if base is None:
            self._img_cache[key] = None
            return None

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
        key = (f"{atlas.atlas_path}::{tile.id}", tile_px, True)
        cached = self._img_cache.get(key)
        if cached is not None:
            return cached

        crop = atlas.crop(tile).convert("RGBA")
        im = crop.resize((tile_px, tile_px), resample=Image.NEAREST)
        self._img_cache[key] = im
        return im

    # ---------- masks ----------
    @staticmethod
    def _mask4(grid: np.ndarray, y: int, x: int, fn: Callable[[int], bool]) -> int:
        h, w = grid.shape
        m = 0
        if y > 0 and fn(int(grid[y - 1, x])):
            m |= 1
        if x < w - 1 and fn(int(grid[y, x + 1])):
            m |= 2
        if y < h - 1 and fn(int(grid[y + 1, x])):
            m |= 4
        if x > 0 and fn(int(grid[y, x - 1])):
            m |= 8
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
                    if y > 0:
                        d = min(d, dist[y - 1, x] + 1)
                    if x > 0:
                        d = min(d, dist[y, x - 1] + 1)
                    dist[y, x] = d
            for y in range(H - 1, -1, -1):
                for x in range(W - 1, -1, -1):
                    d = dist[y, x]
                    if y < H - 1:
                        d = min(d, dist[y + 1, x] + 1)
                    if x < W - 1:
                        d = min(d, dist[y, x + 1] + 1)
                    dist[y, x] = d
        return dist

    def _classify_forest_sprites(self) -> Tuple[List[Sprite], List[Sprite]]:
        trees: List[Sprite] = []
        shrubs: List[Sprite] = []

        for sp in self.forest_sprites_all:
            if sp.tiles_w > 1 or sp.tiles_h > 1:
                trees.append(sp)
            else:
                try:
                    with Image.open(sp.path) as img:
                        _, h = img.size
                        if h > self.tile_raw_px * 1.5:
                            trees.append(sp)
                        else:
                            shrubs.append(sp)
                except Exception:
                    shrubs.append(sp)

        if not trees and shrubs:
            trees = shrubs[:]

        return trees, shrubs

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
        force_fit: bool = False,
    ) -> None:
        h, w = occ.shape
        target = int(h * w * density)
        placed = 0
        if not sprites:
            return

        sprites = [sp for sp in sprites if sp.path.exists()]
        if not sprites:
            return

        for _ in range(attempts):
            if placed >= target:
                break

            sp = rng.choice(sprites)

            # --- LÓGICA DE TAMANHO ---
            if force_fit:
                # Limita a ocupar no MÁXIMO 2x2 no grid (Clamping)
                occ_w = min(sp.tiles_w, 2)
                occ_h = min(sp.tiles_h, 2)
            else:
                occ_w = sp.tiles_w
                occ_h = sp.tiles_h

            y = rng.randrange(0, h - (occ_h - 1))
            x = rng.randrange(0, w - (occ_w - 1))

            base_y = y + occ_h - 1
            if base_y >= h or not allowed_anchor[base_y, x]:
                continue

            if occ[y : y + occ_h, x : x + occ_w].any():
                continue

            occ[y : y + occ_h, x : x + occ_w] = True

            try:
                base_im = Image.open(sp.path).convert("RGBA")
            except Exception:
                continue

            if force_fit:
                # Redimensiona para caber EXATAMENTE no espaço reservado (max 2x2)
                target_w = int(occ_w * tile_px * 0.95)
                target_h = int(occ_h * tile_px * 0.95)
                im_final = base_im.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                im_final = base_im
                if im_final.height > tile_px:
                    scale = tile_px / float(im_final.height) * 0.8
                    nw = max(1, int(im_final.width * scale))
                    nh = max(1, int(im_final.height * scale))
                    im_final = im_final.resize((nw, nh), Image.Resampling.LANCZOS)

            im_final = self._add_sprite_shadow(im_final)

            area_w = occ_w * tile_px
            area_h = occ_h * tile_px
            off_x = (area_w - im_final.width) // 2
            off_y = (area_h - im_final.height) // 2

            if force_fit:
                off_y = area_h - im_final.height - (tile_px // 8)

            draw_x = x * tile_px + off_x
            draw_y = y * tile_px + off_y

            canvas.alpha_composite(im_final, (draw_x, draw_y))
            placed += 1

    # ---------- biome grids ----------
    def _make_sea(self, H: int, W: int, rng: random.Random):
        n = value_noise(H, W, rng, scale=7, blur=3)
        water = n < 0.45
        from collections import deque
        vis = np.zeros((H, W), dtype=bool)
        q = deque()
        for x in range(W):
            if water[0, x]: q.append((0, x)); vis[0, x] = True
            if water[H - 1, x]: q.append((H - 1, x)); vis[H - 1, x] = True
        for y in range(H):
            if water[y, 0]: q.append((y, 0)); vis[y, 0] = True
            if water[y, W - 1]: q.append((y, W - 1)); vis[y, W - 1] = True
        while q:
            y, x = q.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and water[ny, nx] and not vis[ny, nx]:
                    vis[ny, nx] = True
                    q.append((ny, nx))
        ocean = vis
        dist = self._distance_to_mask(~ocean)
        sand = (~ocean) & (dist <= 2.5)
        grass = (~ocean) & (~sand)
        return ocean, sand, grass

    def _make_river(self, H: int, W: int, rng: random.Random):
        y = rng.randrange(H // 4, 3 * H // 4)
        x = 0
        path = np.zeros((H, W), dtype=bool)
        while x < W:
            path[y, x] = True
            x += 1
            if rng.random() < 0.55:
                y += rng.choice([-1, 0, 1])
                y = max(1, min(H - 2, y))
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

    def _edge_mask(self, tile_px: int, direction: str, rng: random.Random, band: int = 14) -> Image.Image:
        band = max(6, min(tile_px // 2, band))
        a = np.zeros((tile_px, tile_px), dtype=np.float32)
        if direction == "N":
            for yy in range(tile_px):
                t = 1.0 - min(1.0, yy / float(band))
                a[yy, :] = t
        elif direction == "S":
            for yy in range(tile_px):
                t = 1.0 - min(1.0, (tile_px - 1 - yy) / float(band))
                a[yy, :] = t
        elif direction == "W":
            for xx in range(tile_px):
                t = 1.0 - min(1.0, xx / float(band))
                a[:, xx] = t
        else:  # 'E'
            for xx in range(tile_px):
                t = 1.0 - min(1.0, (tile_px - 1 - xx) / float(band))
                a[:, xx] = t
        noise = np.array([[rng.random() for _ in range(tile_px)] for _ in range(tile_px)], dtype=np.float32)
        noise = _blur2d(noise, iters=1)
        a = np.clip(a * (0.85 + 0.35 * noise), 0.0, 1.0)
        a = np.clip((a - 0.10) / 0.90, 0.0, 1.0)
        return Image.fromarray((a * 255).astype(np.uint8), mode="L")

    def _blend_tiles(self, base: Image.Image, overlay: Image.Image, mask: Image.Image) -> Image.Image:
        out = base.copy()
        out.paste(overlay, (0, 0), mask)
        return out

    def _lighten_tile(self, im: Image.Image, amount: float) -> Image.Image:
        enhancer = ImageEnhance.Brightness(im)
        return enhancer.enhance(amount)

    def _apply_grass_sand_transition(self, grid: np.ndarray, x: int, y: int, tile_px: int, rng: random.Random, base_tile: Image.Image) -> Image.Image:
        H, W = grid.shape
        if int(grid[y, x]) != 3:
            return base_tile
        touches: List[str] = []
        if y > 0 and int(grid[y - 1, x]) in (0, 1): touches.append("N")
        if x < W - 1 and int(grid[y, x + 1]) in (0, 1): touches.append("E")
        if y < H - 1 and int(grid[y + 1, x]) in (0, 1): touches.append("S")
        if x > 0 and int(grid[y, x - 1]) in (0, 1): touches.append("W")
        if not touches:
            return base_tile
        grass_tile = self._crop_atlas_tile(self.grass_atlas, self.grass_atlas.pick_any(rng), tile_px)
        out = base_tile
        for d in touches:
            mask = self._edge_mask(tile_px, d, rng, band=max(10, tile_px // 5))
            out = self._blend_tiles(out, grass_tile, mask)
        return out

    # ---------- public API ----------
    def generate(
        self,
        biome: str,
        grid_w: int = 32,
        grid_h: int = 32,
        tile_px: int = 64,
        seed: int = 0,
    ) -> Image.Image:
        rng = random.Random(seed)
        biome = biome.lower().strip()

        grid = np.zeros((grid_h, grid_w), dtype=np.int8)
        dist_to_land: Optional[np.ndarray] = None
        damp_mask: Optional[np.ndarray] = None

        if biome == "center_lake":
            grid[:, :] = 0
            margin_x = max(4, grid_w // 5)
            margin_y = max(4, grid_h // 5)
            for yy in range(margin_y, grid_h - margin_y):
                for xx in range(margin_x, grid_w - margin_x):
                    grid[yy, xx] = 5
            dist_to_land = self._distance_to_mask(~(grid == 5))
        elif biome == "sea":
            ocean, sand, grass = self._make_sea(grid_h, grid_w, rng)
            grid[ocean] = 5; grid[sand] = 3; grid[grass] = 0
            dist_to_land = self._distance_to_mask(~(grid == 5))
        elif biome == "river":
            river, sand, grass = self._make_river(grid_h, grid_w, rng)
            grid[river] = 5; grid[sand] = 3; grid[grass] = 0
            dist_to_land = self._distance_to_mask(~(grid == 5))
        elif biome == "forest":
            dark, clear = self._make_forest(grid_h, grid_w, rng)
            grid[dark] = 1; grid[clear] = 0
        elif biome == "prairie":
            grass, dirt = self._make_prairie(grid_h, grid_w, rng)
            grid[grass] = 0; grid[dirt] = 2
        elif biome == "dirt":
            dirt, grass = self._make_dirt(grid_h, grid_w, rng)
            grid[dirt] = 2; grid[grass] = 0
        elif biome == "desert":
            sand, damp = self._make_desert(grid_h, grid_w, rng)
            grid[sand] = 3; damp_mask = damp.astype(np.bool_)
        elif biome == "cave":
            # [ARENA GENERATION]
            # 4 = wall, 2 = light_dirt floor (predominant), 3 = sand accents near the wall.
            # Goal:
            # - Walls surround the arena.
            # - Inside is mostly light_dirt.
            # - A 1-tile sand band hugs the wall on the inside ("na parte marrom sempre sand perto").
            grid.fill(4)

            cave_margin = max(2, min(6, grid_w // 6))

            # Interior base: light_dirt
            for y in range(cave_margin, grid_h - cave_margin):
                for x in range(cave_margin, grid_w - cave_margin):
                    grid[y, x] = 2

            # Sand band hugging the wall (inside ring)
            inner0 = cave_margin
            inner1 = grid_h - cave_margin - 1
            innerL = cave_margin
            innerR = grid_w - cave_margin - 1
            for x in range(innerL, innerR + 1):
                grid[inner0, x] = 3
                grid[inner1, x] = 3
            for y in range(inner0, inner1 + 1):
                grid[y, innerL] = 3
                grid[y, innerR] = 3

            # A few subtle sand patches in the interior (very low), keep predominance of light_dirt.
            # Avoid near-wall band so the ring stays clean.
            for _ in range(max(1, (grid_w * grid_h) // 180)):
                yy = rng.randint(cave_margin + 2, grid_h - cave_margin - 3)
                xx = rng.randint(cave_margin + 2, grid_w - cave_margin - 3)
                if rng.random() < 0.35:
                    grid[yy, xx] = 3
        else:
            grid[:, :] = 0

        canvas = Image.new("RGBA", (grid_w * tile_px, grid_h * tile_px), (0, 0, 0, 0))

        def blit_tile(xx: int, yy: int, im: Image.Image) -> None:
            canvas.alpha_composite(im, (xx * tile_px, yy * tile_px))

        # --- Ground layer ---
        for y in range(grid_h):
            for x in range(grid_w):
                t = int(grid[y, x])

                if t == 5:
                    def neigh_is_land(v: int) -> bool: return v != 5
                    land_mask = self._mask4(grid, y, x, neigh_is_land)
                    if land_mask > 0:
                        water_im: Optional[Image.Image] = None
                        edge_ref = self.water_grass.pick_mask(1, rng)
                        corner_ref = self.water_grass.pick_mask(3, rng)
                        base_edge = self._load_resized(edge_ref, tile_px, force_tile=True) if edge_ref else None
                        base_corner = self._load_resized(corner_ref, tile_px, force_tile=True) if corner_ref else None
                        def rot(im: Optional[Image.Image], angle: int) -> Optional[Image.Image]:
                            if im is None: return None
                            return im.rotate(angle, resample=Image.NEAREST)
                        if land_mask == 1: water_im = rot(base_edge, 0)
                        elif land_mask == 4: water_im = rot(base_edge, 180)
                        elif land_mask == 2: water_im = rot(base_edge, 270)
                        elif land_mask == 8: water_im = rot(base_edge, 90)
                        elif land_mask == 3: water_im = rot(base_corner, 0)
                        elif land_mask == 9: water_im = rot(base_corner, 90)
                        elif land_mask == 12: water_im = rot(base_corner, 180)
                        elif land_mask == 6: water_im = rot(base_corner, 270)
                        if water_im is None:
                            tilep = self.water_grass.pick_mask(land_mask, rng)
                            if tilep: water_im = self._load_resized(tilep, tile_px, force_tile=True)
                            else:
                                tilep = self.water_core.pick_plain(rng)
                                water_im = self._load_resized(tilep, tile_px, force_tile=True)
                        if water_im is None: water_im = Image.new("RGBA", (tile_px, tile_px), (0, 0, 0, 0))
                        blit_tile(x, y, water_im)
                    else:
                        plain_lst = self.water_core.plain
                        prefer = "deep" if (dist_to_land is not None and dist_to_land[y, x] > 2) else "shallow"
                        def _n(p: Path) -> str: return str(p).lower()
                        cand = []
                        if plain_lst:
                            cand = [p for p in plain_lst if ("deep" in _n(p) if prefer == "deep" else "shallow" in _n(p))]
                        tilep = rng.choice(cand) if cand else self.water_core.pick_plain(rng)
                        im = self._load_resized(tilep, tile_px, force_tile=True)
                        if im is None: im = Image.new("RGBA", (tile_px, tile_px), (0, 0, 0, 0))
                        blit_tile(x, y, im)

                elif t == 1:
                    tile = self.dark_grass_atlas.pick_any(rng)
                    blit_tile(x, y, self._crop_atlas_tile(self.dark_grass_atlas, tile, tile_px))

                elif t == 2:
                    tile = self.light_dirt_atlas.pick_any(rng)
                    canvas.alpha_composite(self._crop_atlas_tile(self.light_dirt_atlas, tile, tile_px), (x * tile_px, y * tile_px))

                elif t == 3: # Sand
                    # [SAND LOGIC] Sempre usa sand_atlas para t=3
                    tile = self.sand_atlas.pick_any(rng)
                    sand_im = self._crop_atlas_tile(self.sand_atlas, tile, tile_px)
                    
                    if sand_im is None: sand_im = Image.new("RGBA", (tile_px, tile_px), (0, 0, 0, 0))
                    
                    # Se for deserto ou caverna, não faz transição de grama.
                    if biome != "desert" and biome != "cave":
                         sand_im = self._apply_grass_sand_transition(grid, x, y, tile_px, rng, sand_im)
                    
                    blit_tile(x, y, sand_im)

                elif t == 4:  # Rock / Wall
                    # Desenha um chão base para não ficar buraco
                    base_tile = self.light_dirt_atlas.pick_any(rng)
                    base_im = self._crop_atlas_tile(self.light_dirt_atlas, base_tile, tile_px)
                    canvas.alpha_composite(base_im, (x * tile_px, y * tile_px))

                    # Verifica se toca no chão (na caverna o chão agora é t=3 Sand)
                    def is_floor(v): return v == 3 or v == 2
                    mask = self._mask4(grid, y, x, is_floor)
                    
                    # [WALL LOGIC] Se for borda (mask > 0), usa dirt_rock_edge
                    # A lógica do dirt_rock_edge é: Rock (pedra) fora, Dirt (terra/marrom) na borda tocando o vizinho.
                    # Isso satisfaz: "Pedra virada parede e terra virada interior".
                    if mask > 0 and self.dirt_rock_edge.masks:
                        tilep = self.dirt_rock_edge.pick_mask(mask, rng)
                        img = self._load_resized(tilep, tile_px, force_tile=True)

                        # --- Orientation fix for cave arena walls ---
                        # The dirt_rock_edge source tiles are "horizontal" (dirt on top, rock on bottom).
                        # We rotate them so the brown/dirt side always faces the arena interior.
                        # - If interior is SOUTH (floor below): this is the TOP wall -> rotate 180
                        # - If interior is NORTH (floor above): this is the BOTTOM wall -> rotate 0
                        # - If interior is EAST  (floor right): this is the LEFT wall -> rotate CCW ("to the left")
                        # - If interior is WEST  (floor left):  this is the RIGHT wall -> rotate CW  ("to the right")
                        rot_angle = 0
                        if y < grid_h - 1 and is_floor(int(grid[y + 1, x])):  # interior below
                            rot_angle = 180
                        elif y > 0 and is_floor(int(grid[y - 1, x])):  # interior above
                            rot_angle = 0
                        elif x < grid_w - 1 and is_floor(int(grid[y, x + 1])):  # interior right
                            rot_angle = 90  # CCW
                        elif x > 0 and is_floor(int(grid[y, x - 1])):  # interior left
                            rot_angle = -90  # CW

                        if img is not None and rot_angle != 0:
                            img = img.rotate(rot_angle, resample=Image.NEAREST, expand=False)

                        canvas.alpha_composite(img, (x * tile_px, y * tile_px))
                    else:
                        tilep = self.rock_floor.pick_plain(rng)
                        canvas.alpha_composite(self._load_resized(tilep, tile_px, force_tile=True), (x * tile_px, y * tile_px))

        # --- Overlay layer ---
        occ = np.zeros((grid_h, grid_w), dtype=bool)
        is_land = (grid != 5)

        # Rocks gerais
        rock_density = 0.030 if biome == "sea" else 0.020
        # Na caverna, não usamos rocks_sprites, usamos apenas cave_sprites
        if biome != "cave":
             self._place_sprites(canvas, rng, self.rocks_sprites, occ, tile_px, attempts=1600, density=rock_density, allowed_anchor=is_land)

        if biome == "forest":
            is_light_grass = grid == 0; is_dark_grass = grid == 1; is_any_grass = is_light_grass | is_dark_grass
            self._place_sprites(canvas, rng, self.forest_trees, occ, tile_px, attempts=8000, density=0.25, allowed_anchor=is_any_grass, force_fit=True)
            self._place_sprites(canvas, rng, self.forest_shrubs, occ, tile_px, attempts=4000, density=0.10, allowed_anchor=is_dark_grass)
            self._place_sprites(canvas, rng, self.flower_sprites, occ, tile_px, attempts=5000, density=0.15, allowed_anchor=is_light_grass)
            self._place_sprites(canvas, rng, self.foliage_sprites, occ, tile_px, attempts=3000, density=0.05, allowed_anchor=is_any_grass)

        elif biome == "prairie":
            is_grass = grid == 0
            self._place_sprites(canvas, rng, self.flower_sprites, occ, tile_px, attempts=4500, density=0.08, allowed_anchor=is_grass)
            self._place_sprites(canvas, rng, self.foliage_sprites, occ, tile_px, attempts=3500, density=0.05, allowed_anchor=is_grass)
            self._place_sprites(canvas, rng, self.forest_shrubs, occ, tile_px, attempts=1200, density=0.01, allowed_anchor=is_grass)

        elif biome == "dirt":
            is_grass = grid == 0
            self._place_sprites(canvas, rng, self.foliage_sprites, occ, tile_px, attempts=2000, density=0.02, allowed_anchor=is_grass)

        elif biome == "desert":
            self._place_sprites(canvas, rng, self.misc_sprites, occ, tile_px, attempts=1600, density=0.015, allowed_anchor=is_land)

        elif biome == "cave":
            # Use ONLY cave_overlay sprites and keep them focused in the CENTER of the arena,
            # not on the sand band near the walls.
            is_floor_any = (grid == 2) | (grid == 3)

            # Build a "center-only" mask (avoid borders + avoid the sand ring).
            # We reconstruct the margin with the same rule used above.
            cave_margin = max(2, min(6, grid_w // 6))
            center = np.zeros((grid_h, grid_w), dtype=bool)
            y0 = cave_margin + 2
            y1 = grid_h - cave_margin - 3
            x0 = cave_margin + 2
            x1 = grid_w - cave_margin - 3
            if y1 >= y0 and x1 >= x0:
                center[y0:y1+1, x0:x1+1] = True

            # Exclude sand ring (t==3) so decorations are not hugging walls.
            allowed_anchor = is_floor_any & center & (grid != 3)

            # Separate 1x1 decals (low alpha coverage) from 1x1 boulders to avoid "rock soup".
            if not hasattr(self, "_alpha_ratio_cache"):
                self._alpha_ratio_cache = {}

            def alpha_ratio(sp: Sprite) -> float:
                k = str(sp.path)
                if k in self._alpha_ratio_cache:
                    return self._alpha_ratio_cache[k]
                try:
                    im = Image.open(sp.path).convert("RGBA")
                    a = np.array(im.split()[-1])
                    r = float((a > 10).mean())
                except Exception:
                    r = 0.0
                self._alpha_ratio_cache[k] = r
                return r

            cave_1x1 = [s for s in self.cave_sprites if s.tiles_w == 1 and s.tiles_h == 1]
            cave_big = [s for s in self.cave_sprites if s.tiles_w > 1 or s.tiles_h > 1]

            cave_decal: List[Sprite] = []
            cave_boulder1x1: List[Sprite] = []
            for s in cave_1x1:
                if alpha_ratio(s) < 0.22:
                    cave_decal.append(s)
                else:
                    cave_boulder1x1.append(s)

            # Very reduced quantities (center only).
            self._place_sprites(canvas, rng, cave_big, occ, tile_px,
                                attempts=14, density=0.004, allowed_anchor=allowed_anchor, force_fit=True)
            self._place_sprites(canvas, rng, cave_boulder1x1, occ, tile_px,
                                attempts=40, density=0.006, allowed_anchor=allowed_anchor, force_fit=True)
            self._place_sprites(canvas, rng, cave_decal, occ, tile_px,
                                attempts=120, density=0.012, allowed_anchor=allowed_anchor, force_fit=True)

        elif biome == "center_lake":
            is_land = (grid == 0) | (grid == 1) | (grid == 2)
            self._place_sprites(canvas, rng, self.flower_sprites, occ, tile_px, attempts=4000, density=0.12, allowed_anchor=is_land)
            self._place_sprites(canvas, rng, self.misc_sprites, occ, tile_px, attempts=1500, density=0.03, allowed_anchor=is_land)
            self._place_sprites(canvas, rng, self.forest_trees, occ, tile_px, attempts=300, density=0.01, allowed_anchor=is_land)
            self._place_sprites(canvas, rng, self.forest_shrubs, occ, tile_px, attempts=800, density=0.02, allowed_anchor=is_land)

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
