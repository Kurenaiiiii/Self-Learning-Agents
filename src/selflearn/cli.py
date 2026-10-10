"""selflearn CLI: chat, memories, eval, doctor."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

from selflearn import __version__
from selflearn.config import Config, load_config
from selflearn.doctor import run_doctor

if TYPE_CHECKING:
    from selflearn.agent import MemoryAgent


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with all subcommands."""
    p = argparse.ArgumentParser(prog="selflearn", description="Self-Learning Agents")
    p.add_argument("--version", action="store_true", help="print version")
    p.add_argument("--config", default=None, help="optional JSON config file")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("chat", help="chat with the memory agent")
    c.add_argument("--user", default="default", help="user id (memory owner)")
    c.add_argument("--namespace", default="default", help="memory namespace")
    c.add_argument("--message", default=None, help="single message (else interactive)")
    c.add_argument("--no-learn", action="store_true", help="disable post-turn learning")
    c.add_argument("--no-tools", action="store_true", help="disable agent tools")

    m = sub.add_parser("memories", help="manage memories")
    msub = m.add_subparsers(dest="mcmd", required=True)
    ml = msub.add_parser("list", help="list memories")
    ml.add_argument("--user", required=True)
    ml.add_argument("--namespace", default="default")
    ml.add_argument("--type", default=None)
    ml.add_argument("--limit", type=int, default=50)
    ms = msub.add_parser("search", help="search memories")
    ms.add_argument("--user", required=True)
    ms.add_argument("--namespace", default="default")
    ms.add_argument("query")
    ms.add_argument("--top-k", type=int, default=5)
    mf = msub.add_parser("forget", help="hard-delete ALL of a user's data")
    mf.add_argument("--user", required=True)
    mf.add_argument("--yes", action="store_true", help="skip confirmation")
    me = msub.add_parser("export", help="export a user's data as JSON")
    me.add_argument("--user", required=True)
    me.add_argument("--out", default=None, help="output file (else stdout)")

    e = sub.add_parser("eval", help="run the memory benchmark")
    e.add_argument("--config-name", default="hybrid", help="memory strategy to evaluate")
    e.add_argument("--scenarios", default=None, help="scenario file (else built-in)")
    e.add_argument("--runs", type=int, default=3)
    e.add_argument("--out", default=None, help="results JSON file")
    e.add_argument("--list-strategies", action="store_true")

    sub.add_parser("doctor", help="check config and connectivity")
    return p


def _agent_for(cfg: Config, args: argparse.Namespace) -> MemoryAgent:
    from selflearn.agent import MemoryAgent
    from selflearn.llm.factory import provider_from_config
    from selflearn.store import MemoryStore

    return MemoryAgent(
        llm=provider_from_config(cfg),
        store=MemoryStore(cfg.db_path),
        model=cfg.model,
        user=args.user,
        namespace=args.namespace,
        learn=not args.no_learn,
        enable_tools=not args.no_tools,
    )


def cmd_chat(cfg: Config, args: argparse.Namespace) -> int:
    """Interactive or single-shot chat."""
    try:
        agent = _agent_for(cfg, args)
    except ValueError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.message:
        print(agent.chat(args.message).text)
        return 0
    print(f"Chatting as {args.user} (model {cfg.model}). Empty line quits.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            return 0
        print("ai>", agent.chat(line).text)


def cmd_memories(cfg: Config, args: argparse.Namespace) -> int:
    """Memory list/search/forget/export."""
    from selflearn.memory import Retriever, export_user_json, forget_me
    from selflearn.store import MemoryStore

    store = MemoryStore(cfg.db_path)
    if args.mcmd == "list":
        for mem in store.list(
            user=args.user, namespace=args.namespace, type=args.type, limit=args.limit
        ):
            print(f"{mem.id} [{mem.type}] (imp {mem.importance:.2f}) {mem.text}")
        return 0
    if args.mcmd == "search":
        for row in Retriever(store, user=args.user, namespace=args.namespace).explain(
            args.query, top_k=args.top_k
        ):
            print(f"{row['score']:.3f} [{row['type']}] {row['text']}")
        return 0
    if args.mcmd == "forget":
        if not args.yes:
            answer = input(f"Delete ALL data for user {args.user!r}? Type YES: ")
            if answer.strip() != "YES":
                print("aborted")
                return 1
        print(f"removed {forget_me(store, user=args.user)} rows")
        return 0
    if args.mcmd == "export":
        payload = export_user_json(store, user=args.user)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(payload)
            print(f"wrote {args.out}")
        else:
            print(payload)
        return 0
    return 2


def cmd_eval(cfg: Config, args: argparse.Namespace) -> int:
    """Run the benchmark harness."""
    from selflearn.bench import STRATEGIES, run_benchmark

    if args.list_strategies:
        print("strategies:", ", ".join(sorted(STRATEGIES)))
        return 0
    report = run_benchmark(
        cfg, strategy=args.config_name, runs=args.runs, scenarios_path=args.scenarios
    )
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.cmd is None:
        build_parser().print_help()
        return 2
    cfg = load_config(args.config)
    if args.cmd == "doctor":
        print(run_doctor(cfg))
        return 0
    if args.cmd == "chat":
        return cmd_chat(cfg, args)
    if args.cmd == "memories":
        return cmd_memories(cfg, args)
    if args.cmd == "eval":
        return cmd_eval(cfg, args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
