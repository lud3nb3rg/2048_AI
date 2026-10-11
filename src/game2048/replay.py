"""Record one game and save it as an interactive HTML replay, and optionally a GIF.

    uv run replay-2048 --agent ppo --model checkpoints/ppo_cnn64_norm/final.zip --open
    uv run replay-2048 --agent ppo --model ... --best-of 200 --gif docs/replay.gif
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np

from game2048.agents import AGENTS, Agent, CornerAgent, PPOAgent
from game2048.env import Game2048Env
from game2048.evaluate import evaluate, evaluate_batched
from game2048.render import _BACKGROUND, _DARK_TEXT, _LIGHT_TEXT, _TILE_COLOURS, render_rgb

_TEMPLATE = Path(__file__).with_name("replay_template.html")


def record_game(agent: Agent, seed: int | None = None, size: int = 4) -> list[dict[str, Any]]:
    """Play one game and return a frame per position.

    Frame ``i`` holds the board *before* move ``i``, the move the agent chose from
    it (``a``), the policy's move probabilities (``p``, if the agent exposes
    ``action_probs``), the legal-move mask (``m``), the score so far (``s``), the
    points the previous move earned (``g``), and where the last tile spawned
    (``sp``, a flat cell index). The final frame is the game-over board.
    """
    env = Game2048Env(size=size)
    obs, info = env.reset(seed=seed)
    frames: list[dict[str, Any]] = []
    gain, spawn, done = 0, None, False
    while True:
        frame = {
            "b": obs.ravel().tolist(),
            "s": info["score"],
            "m": info["action_mask"].astype(int).tolist(),
            "g": gain,
            "sp": spawn,
            "a": None,
            "p": None,
        }
        frames.append(frame)
        if done:
            return frames
        action = int(agent(obs, info))
        frame["a"] = action
        if hasattr(agent, "action_probs"):
            probs = agent.action_probs(obs, info["action_mask"])
            frame["p"] = [round(float(p), 4) for p in probs]
        afterstate = env.game.peek(action).board
        prev_score = info["score"]
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        gain = info["score"] - prev_score
        new_cells = np.flatnonzero((afterstate.ravel() == 0) & (obs.ravel() != 0))
        spawn = int(new_cells[0]) if info["valid_move"] and new_cells.size else None


def write_html(frames: list[dict[str, Any]], path: Path, meta: dict[str, Any], size: int) -> None:
    data = {
        "meta": meta,
        "size": size,
        "frames": frames,
        "palette": ["rgb({},{},{})".format(*c) for c in _TILE_COLOURS],
        "darkText": "rgb({},{},{})".format(*_DARK_TEXT),
        "lightText": "rgb({},{},{})".format(*_LIGHT_TEXT),
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = _TEMPLATE.read_text(encoding="utf-8").replace("__REPLAY_DATA__", payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_gif(
    frames: list[dict[str, Any]], path: Path, size: int, fps: float = 20, tile: int = 64,
    hold_end_ms: int = 3000,
) -> None:
    """Save the game as an animated GIF (needs Pillow: it's in the dev dependency group)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("--gif needs Pillow: uv add --dev pillow") from exc

    gap = max(2, tile // 10)
    # The renderer only paints a fixed set of flat colours, so every frame maps
    # exactly onto one shared palette: crisp, no dithering, and small.
    colours = np.array(sorted({*_TILE_COLOURS, _BACKGROUND, _DARK_TEXT, _LIGHT_TEXT}))
    keys = (colours[:, 0] << 16) | (colours[:, 1] << 8) | colours[:, 2]
    order = np.argsort(keys)
    keys, colours = keys[order], colours[order]

    def to_indexed(rgb: np.ndarray) -> Image.Image:
        packed = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
        img = Image.fromarray(np.searchsorted(keys, packed).astype(np.uint8), mode="P")
        img.putpalette(colours.astype(np.uint8).ravel().tolist())
        return img

    images = [
        to_indexed(render_rgb(np.array(f["b"], dtype=np.uint8).reshape(size, size), tile, gap))
        for f in frames
    ]
    durations = [round(1000 / fps)] * len(images)
    durations[-1] = hold_end_ms
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False
    )


def _best_seed(agent: Agent, episodes: int, seed: int, size: int) -> int:
    if isinstance(agent, PPOAgent):
        result = evaluate_batched(agent, episodes, seed=seed, size=size)
    else:
        result = evaluate(Game2048Env(size=size), agent, episodes, seed=seed)
    best = int(np.argmax(result.scores))
    print(
        f"best of {episodes}: seed {seed + best} (score {result.scores[best]:,}, "
        f"max tile {result.max_tiles[best]}); mean was {np.mean(result.scores):,.0f}"
    )
    return seed + best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=sorted(AGENTS), default="ppo")
    parser.add_argument("--model", help="path to a trained checkpoint (required for --agent ppo)")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true", help="sample moves (ppo only)")
    parser.add_argument(
        "--best-of", type=int, metavar="N",
        help="play N games (seeds SEED..SEED+N-1) and replay the highest-scoring one",
    )
    parser.add_argument("--out", type=Path, help="HTML output path (default: replays/<name>.html)")
    parser.add_argument("--open", action="store_true", help="open the replay in a browser")
    parser.add_argument("--gif", type=Path, help="also save the game as an animated GIF here")
    parser.add_argument("--gif-fps", type=float, default=20, help="GIF moves per second")
    parser.add_argument("--gif-tile", type=int, default=64, help="GIF tile size in pixels")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (ppo only)")
    args = parser.parse_args(argv)

    if args.agent == "ppo":
        if not args.model:
            parser.error("--agent ppo requires --model PATH")
        import torch

        torch.set_num_threads(args.threads)
        agent = PPOAgent(
            args.model, seed=args.seed, size=args.size, deterministic=not args.stochastic
        )
        deterministic = not args.stochastic
    else:
        agent = AGENTS[args.agent](seed=args.seed)
        deterministic = isinstance(agent, CornerAgent)

    seed = args.seed
    if args.best_of:
        if not deterministic:
            parser.error("--best-of needs a deterministic agent (ppo without --stochastic, or corner)")
        seed = _best_seed(agent, args.best_of, args.seed, args.size)

    frames = record_game(agent, seed=seed, size=args.size)
    last = frames[-1]
    max_tile = 1 << max(last["b"])
    name = Path(args.model).stem if args.model else args.agent
    meta = {
        "agent": args.agent + (" (stochastic)" if args.stochastic else ""),
        "model": str(args.model) if args.model else None,
        "seed": seed,
        "score": last["s"],
        "maxTile": max_tile,
        "moves": sum(1 for f in frames if f["a"] is not None and f["m"][f["a"]]),
    }
    out = args.out or Path("replays") / f"{name}_seed{seed}.html"
    write_html(frames, out, meta, args.size)
    print(f"replay: {out}  (score {meta['score']:,}, max tile {max_tile}, {meta['moves']} moves)")
    if args.gif:
        write_gif(frames, args.gif, args.size, fps=args.gif_fps, tile=args.gif_tile)
        print(f"gif: {args.gif}  ({args.gif.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
