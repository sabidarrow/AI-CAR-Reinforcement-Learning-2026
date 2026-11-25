"""Run the visual lab, train headlessly, or compare policies on held-out seeds."""
import argparse
import csv
import json
import statistics
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from loop_sim.core import QAgent, World, heuristic

ROOT = Path(__file__).resolve().parent


class Session:
    def __init__(self, args):
        self.args = args
        self.agent = QAgent(args.seed + 1)
        if args.model.exists():
            self.agent.load(args.model)
        self.world = World(args.seed, not args.no_v2x)
        self.history = []
        self.state = self.world.observe()
        self.args.output.mkdir(parents=True, exist_ok=True)
        self.log = self.args.output / "episodes.csv"

    def advance(self, count, training=True):
        for _ in range(count):
            if self.world.done:
                self.state = self.world.reset()
            action = self.agent.act(self.state, training)
            next_state, reward, done = self.world.step(action)
            if training:
                self.agent.learn(self.state, action, reward, next_state, done)
            self.state = next_state
            if done:
                if training:
                    self.agent.episodes += 1
                row = {"episode": self.agent.episodes, "training": training,
                       "v2x": self.world.connected, "seed": self.args.seed,
                       "reward": round(self.world.total_reward, 3),
                       "distance_m": round(self.world.distance, 2),
                       "steps": self.world.steps, "crash": self.world.reason != "timeout",
                       "reason": self.world.reason, "near_misses": self.world.near_misses,
                       "epsilon": round(self.agent.epsilon if training else 0, 4)}
                self.history.append(row)
                self.history = self.history[-200:]
                new = not self.log.exists() or self.log.stat().st_size == 0
                with self.log.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
                if training and self.agent.episodes % 25 == 0:
                    self.agent.save(self.args.model)
                # Leave the crash visible for one rendered frame.
                break

    def snapshot(self):
        return {"world": self.world.snapshot(), "history": self.history,
                "episodes": self.agent.episodes, "epsilon": 0 if self.args.watch else self.agent.epsilon,
                "training": not self.args.watch,
                "states": len(self.agent.q), "updates": self.agent.updates,
                "td_error": self.agent.last_error}


def evaluate(args, agent):
    rows = []
    for connected in (False, True):
        for policy in ("random", "heuristic", "learned"):
            for i in range(args.episodes):
                world = World(args.seed + 100_000 + i, connected)
                import random
                rng = random.Random(args.seed + i)
                state = world.observe()
                while not world.done:
                    action = rng.randrange(5) if policy == "random" else heuristic(world) if policy == "heuristic" else agent.act(state, False)
                    state, _, _ = world.step(action)
                rows.append({"policy": policy, "v2x": connected, "seed": args.seed+100_000+i,
                             "reward": world.total_reward, "distance_m": world.distance,
                             "crash": world.reason != "timeout", "reason": world.reason,
                             "near_misses": world.near_misses})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for connected in (False, True):
        for policy in ("random", "heuristic", "learned"):
            group = [r for r in rows if r["v2x"] == connected and r["policy"] == policy]
            distance = sum(r["distance_m"] for r in group)
            summary.append({"policy": policy, "v2x": connected, "episodes": len(group),
                "mean_reward": round(statistics.mean(r["reward"] for r in group), 2),
                "mean_distance_m": round(distance/len(group), 2),
                "crash_rate": round(statistics.mean(r["crash"] for r in group), 3),
                "crashes_per_km": round(sum(r["crash"] for r in group)/(distance/1000), 3) if distance else None})
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def serve(session, port, open_browser):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, mime="application/json", status=200):
            payload = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self.reply((ROOT / "loop_sim" / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.reply(session.snapshot())
            else:
                self.reply({"error": "Not found"}, status=404)

        def do_POST(self):
            if self.headers.get("Origin") not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                self.reply({"error": "Local dashboard requests only"}, status=403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if self.path == "/api/step":
                    session.advance(max(1, min(500, int(data.get("count", 1)))), not session.args.watch)
                elif self.path == "/api/save":
                    if not session.args.watch:
                        session.agent.save(session.args.model)
                else:
                    self.reply({"error": "Not found"}, status=404)
                    return
                self.reply(session.snapshot())
            except (ValueError, TypeError) as e:
                self.reply({"error": str(e)}, status=400)

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Driving lab: {url}\nCtrl+C to stop. Model: {session.args.model}", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if not session.args.watch:
            session.agent.save(session.args.model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Train without graphics")
    mode.add_argument("--evaluate", action="store_true", help="Compare three policies, with/without V2X")
    mode.add_argument("--watch", action="store_true", help="Show frozen policy; disable learning/exploration")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-v2x", action="store_true")
    parser.add_argument("--model", type=Path, default=ROOT / "runs" / "q_policy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--browser", action="store_true", help="Use the optional browser UI instead of the desktop window")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if (args.watch or args.evaluate) and not args.model.exists():
        parser.error("Train a model first or supply --model PATH")
    session = Session(args)
    if args.evaluate:
        evaluate(args, session.agent)
    elif args.train:
        target = session.agent.episodes + args.episodes
        try:
            while session.agent.episodes < target:
                session.advance(600)
                if session.agent.episodes % 100 == 0:
                    recent = session.history[-100:]
                    print(f"Episode {session.agent.episodes}: reward={statistics.mean(r['reward'] for r in recent):.1f}, epsilon={session.agent.epsilon:.3f}", flush=True)
        except KeyboardInterrupt:
            print("Training interrupted; saving progress.")
        finally:
            session.agent.save(args.model)
    else:
        if args.browser or args.no_browser:
            serve(session, args.port, not args.no_browser)
        else:
            from loop_sim.desktop import DesktopLab
            DesktopLab(session).run()


if __name__ == "__main__":
    main()
