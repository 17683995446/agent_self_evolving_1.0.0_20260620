"""
统一命令行接口。

提供单一入口控制Agent全部行为。

第一性原理：
- 生物运动皮层统一指挥肢体运动——单一决策中枢协调全部行为
- CLI是用户与系统交互的"运动皮层"，所有操作通过统一入口
- 子命令模式：每个行为一个明确命令，降低认知负荷

设计要点：
- argparse子命令：start/stop/status/health/dashboard/log/version
- 零配置可启动：无参数时使用合理默认值
- 信号处理：Ctrl+C优雅关闭
- 友好输出：彩色状态、进度提示
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from agent_core.common.logger import get_logger
from agent_core.common.constants import LogLevel

# 默认配置
_DEFAULT_ONLINE_INTERVAL = 1.0
_DEFAULT_OFFLINE_INTERVAL = 3600.0
_DEFAULT_DASHBOARD_PORT = 8080
_DEFAULT_ENV = "dev"

# 输出颜色（ANSI转义码）
_COLOR_RESET = "\033[0m"
_COLOR_RED = "\033[31m"
_COLOR_GREEN = "\033[32m"
_COLOR_YELLOW = "\033[33m"
_COLOR_BLUE = "\033[34m"
_COLOR_CYAN = "\033[36m"
_COLOR_BOLD = "\033[1m"

# 是否启用颜色（检测TTY）
_COLOR_ENABLED = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False


def _colorize(text: str, color: str) -> str:
    """添加颜色（非TTY环境不添加）。"""
    if not _COLOR_ENABLED:
        return text
    return f"{color}{text}{_COLOR_RESET}"


def _print_success(msg: str) -> None:
    """打印成功信息。"""
    print(f"{_colorize('[OK]', _COLOR_GREEN)} {msg}")


def _print_error(msg: str) -> None:
    """打印错误信息。"""
    print(f"{_colorize('[ERROR]', _COLOR_RED)} {msg}", file=sys.stderr)


def _print_warning(msg: str) -> None:
    """打印警告信息。"""
    print(f"{_colorize('[WARN]', _COLOR_YELLOW)} {msg}")


def _print_info(msg: str) -> None:
    """打印信息。"""
    print(f"{_colorize('[INFO]', _COLOR_BLUE)} {msg}")


def _load_config(env: str, config_path: Optional[str]) -> Dict[str, Any]:
    """
    加载配置。

    优先级：config_path > env配置 > base配置 > 默认值

    Args:
        env: 环境名（dev/prod/test）。
        config_path: 显式配置文件路径。

    Returns:
        配置字典。
    """
    from agent_core.common.config_validator import ConfigValidator

    config: Dict[str, Any] = {}

    if config_path and os.path.exists(config_path):
        # 显式配置文件优先
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            _print_info(f"Loaded config from {config_path}")
        except ImportError:
            _print_warning("yaml not available, using defaults")
        except OSError as exc:
            _print_error(f"Failed to load config: {exc}")
    else:
        # 使用EnvironmentConfig加载base + env配置
        config_dir = os.environ.get(
            "AGENT_CONFIG_DIR",
            os.path.join(os.getcwd(), "config"),
        )
        try:
            from agent_core.common.environment_config import EnvironmentConfig
            env_config = EnvironmentConfig(config_dir=config_dir)
            # 强制使用指定环境
            if env != env_config.environment:
                # 重新加载指定环境
                os.environ["AGENT_ENV"] = env
                env_config = EnvironmentConfig(config_dir=config_dir)
            config = env_config.all_config
            if config:
                _print_info(f"Loaded {env} config from {config_dir}")
        except Exception as exc:
            _print_warning(f"EnvironmentConfig load failed: {exc}, using defaults")

    # 填充默认值（零配置可启动）
    defaults = _get_default_config()
    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    # 校验配置
    validator = ConfigValidator()
    ok, errors = validator.validate(config)
    if not ok:
        _print_error("Configuration validation failed:")
        for err in errors:
            _print_error(f"  {err}")
        _print_error("Fix the above errors or use --config to specify a valid config file")
        _print_info("Hint: Use 'agent start --dry-run' to check config without starting")
        _print_info("Hint: Use 'agent start' without --config for zero-config defaults")

    return config


def _get_default_config() -> Dict[str, Any]:
    """获取默认配置（零配置启动用）。"""
    return {
        "app_name": "self_evolving_agent",
        "app_version": "1.0.0",
        "log_level": "INFO",
        "online_interval_seconds": _DEFAULT_ONLINE_INTERVAL,
        "offline_interval_seconds": _DEFAULT_OFFLINE_INTERVAL,
        "memory_capacity": 10000,
        "exploration_rate": 0.3,
        "epsilon": 0.2,
        "interrupt_threshold": 0.7,
        "feature_dimensions": 10,
        "low_alpha_init": 0.3,
        "low_alpha_min": 0.05,
        "low_alpha_max": 0.8,
        "low_adaptation_rate": 0.1,
        "mid_window_size": 20,
        "mid_forecast_horizon": 5,
        "high_window_size": 100,
        "high_reversion_strength": 0.3,
        "ensemble_beta": 5.0,
        "cpu_interval_seconds": 1.0,
        "disk_paths": ["/"],
        "log_paths": [],
        "max_log_lines": 100,
        "base_weights": {
            "safety": 0.4,
            "gain": 0.4,
            "efficiency": 0.2,
        },
        "query_page_size": 20,
        "batch_query_limit": 1000,
    }


async def _cmd_start(args: argparse.Namespace) -> int:
    """启动Agent。"""
    from agent_core.memory.agent_main import SelfEvolvingAgent

    config = _load_config(args.env, args.config)

    # CLI参数覆盖配置
    if args.log_level:
        config["log_level"] = args.log_level
    if args.online_interval:
        config["online_interval_seconds"] = args.online_interval
    if args.offline_interval:
        config["offline_interval_seconds"] = args.offline_interval

    # --dry-run：仅验证配置不启动
    if args.dry_run:
        print(_colorize("=== Configuration Check ===", _COLOR_BOLD))
        print(f"  Environment:        {args.env}")
        print(f"  Log Level:          {config.get('log_level', 'INFO')}")
        print(f"  Online Interval:    {config.get('online_interval_seconds', 1.0)}s")
        print(f"  Offline Interval:   {config.get('offline_interval_seconds', 3600.0)}s")
        print(f"  Memory Capacity:    {config.get('memory_capacity', 10000)}")
        print(f"  Exploration Rate:   {config.get('exploration_rate', 0.3)}")
        _print_success("Configuration is valid. Agent is ready to start.")
        _print_info("Run without --dry-run to actually start the Agent.")
        return 0

    _print_info(
        f"Starting Agent (env={args.env}, "
        f"online={config['online_interval_seconds']}s, "
        f"offline={config['offline_interval_seconds']}s)"
    )

    agent = SelfEvolvingAgent()

    # 初始化
    try:
        ok = await agent.initialize(config)
        if not ok:
            _print_error("Agent initialization failed")
            return 1
    except Exception as exc:
        _print_error(f"Initialization error: {exc}")
        return 1

    _print_success("Agent initialized successfully")

    # 信号处理
    stop_event = asyncio.Event()

    def _signal_handler(signum, frame):
        _print_info(f"\nReceived signal {signum}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 启动
    online_interval = config["online_interval_seconds"]
    offline_interval = config["offline_interval_seconds"]

    try:
        await agent.start(online_interval, offline_interval)
        _print_success(
            f"Agent started (dashboard at http://127.0.0.1:{_DEFAULT_DASHBOARD_PORT})"
        )
        _print_info("Press Ctrl+C to stop")

        # 等待停止信号
        await stop_event.wait()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _print_error(f"Runtime error: {exc}")
        return 1
    finally:
        _print_info("Stopping Agent...")
        try:
            await agent.stop()
            _print_success("Agent stopped gracefully")
        except Exception as exc:
            _print_error(f"Error during shutdown: {exc}")

    return 0


async def _cmd_status(args: argparse.Namespace) -> int:
    """查询Agent状态（通过Dashboard API或日志文件）。"""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{args.port}/api/status"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        # Dashboard不可用 — 离线模式
        _print_warning(
            f"Agent is not running (dashboard at port {args.port} not reachable)"
        )
        _print_info("Recent log entries:")

        # 尝试读取最近的日志文件
        log_dir = os.path.join(os.getcwd(), "data", "logs")
        log_file = os.path.join(log_dir, "selfevolvingagent.log")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                # 显示最后10行
                recent = lines[-10:] if len(lines) > 10 else lines
                for line in recent:
                    print(f"  {line.rstrip()}")
                _print_info(f"(showing last {len(recent)} lines from {log_file})")
            except OSError as exc:
                _print_error(f"Failed to read log file: {exc}")
        else:
            _print_info("No log file found. Start the Agent with 'agent start'.")

        _print_info("To start the Agent: agent start")
        return 1
    except Exception as exc:
        _print_error(f"Status query failed: {exc}")
        return 1

    # 格式化输出
    print(_colorize("=== Agent Status ===", _COLOR_BOLD))
    print(f"  Status:       {_colorize(data.get('status', 'unknown'), _COLOR_CYAN)}")
    print(f"  Cycle Count:  {data.get('cycle_count', 0)}")
    print(f"  Interrupts:   {data.get('interrupt_count', 0)}")

    errors = data.get("recent_errors", [])
    if errors:
        print(f"  Recent Errors ({len(errors)}):")
        for err in errors[-3:]:
            print(f"    - {err}")

    rewards = data.get("recent_rewards", [])
    if rewards:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"  Avg Reward:   {avg_reward:.3f} (last {len(rewards)})")

    task = data.get("current_task")
    if task:
        print(f"  Current Task: {task}")

    return 0


async def _cmd_stop(args: argparse.Namespace) -> int:
    """停止运行中的Agent。

    优先通过Dashboard API发送停止请求；
    若Dashboard不可用，回退到查找并终止Agent进程。
    """
    import urllib.request
    import urllib.error
    import subprocess

    # 方式1：通过Dashboard API停止
    url = f"http://127.0.0.1:{args.port}/api/stop"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                _print_success("Agent stop request sent via Dashboard API")
                return 0
    except urllib.error.URLError:
        pass  # Dashboard不可用，回退到进程查找
    except Exception:
        pass  # 其他API错误，回退到进程查找

    # 方式2：查找并终止Agent进程
    _print_info("Dashboard not reachable, trying to find Agent process...")
    try:
        result = subprocess.run(
            ["pgrep", "-f", "agent_core"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        if not pids:
            _print_warning("No running Agent process found")
            return 0

        for pid_str in pids:
            pid = int(pid_str)
            if pid == os.getpid():
                continue  # 不终止自己
            try:
                os.kill(pid, signal.SIGTERM)
                _print_info(f"Sent SIGTERM to PID {pid}")
            except ProcessLookupError:
                pass
            except PermissionError:
                _print_error(f"Permission denied to stop PID {pid}")

        _print_success("Agent stop signal sent")
        return 0
    except FileNotFoundError:
        _print_error("pgrep not available. Use 'kill <pid>' manually.")
        return 1
    except Exception as exc:
        _print_error(f"Failed to stop Agent: {exc}")
        return 1


async def _cmd_health(args: argparse.Namespace) -> int:
    """健康检查。"""
    from agent_core.memory.agent_main import SelfEvolvingAgent

    _print_info("Running health check...")
    agent = SelfEvolvingAgent()
    config = _load_config(args.env, args.config)

    try:
        await agent.initialize(config)
        result = await agent.health_check()
    except Exception as exc:
        _print_error(f"Health check failed: {exc}")
        return 1

    try:
        await agent.stop()
    except Exception:
        pass

    print(_colorize("=== Health Check ===", _COLOR_BOLD))
    overall = result.get("status", "unknown")
    color = _COLOR_GREEN if overall == "healthy" else _COLOR_RED
    print(f"  Overall: {_colorize(overall.upper(), color)}")

    checks = result.get("checks", {})
    for name, check in checks.items():
        status = check.get("status", "unknown") if isinstance(check, dict) else str(check)
        icon = _colorize("[OK]", _COLOR_GREEN) if status == "healthy" else _colorize("[FAIL]", _COLOR_RED)
        print(f"  {icon} {name}: {status}")

    return 0 if overall == "healthy" else 1


async def _cmd_dashboard(args: argparse.Namespace) -> int:
    """启动独立Dashboard服务。"""
    from agent_core.monitoring.dashboard_server import get_dashboard_server

    dashboard = get_dashboard_server()
    _print_info(f"Starting dashboard on port {args.port}...")
    dashboard.start(port=args.port)
    _print_success(f"Dashboard running at http://127.0.0.1:{args.port}")
    _print_info("Press Ctrl+C to stop")

    # 等待信号
    stop_event = asyncio.Event()

    def _signal_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    await stop_event.wait()
    dashboard.stop()
    _print_success("Dashboard stopped")
    return 0


async def _cmd_version(args: argparse.Namespace) -> int:
    """显示版本信息。"""
    from agent_core import __version__
    print(_colorize("Self-Evolving Agent", _COLOR_BOLD))
    print(f"  Version: {__version__}")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  Path:    {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    return 0


async def _cmd_log(args: argparse.Namespace) -> int:
    """查看最近日志。"""
    log_dir = os.path.join(os.getcwd(), "data", "logs")
    log_file = os.path.join(log_dir, "selfevolvingagent.log")

    if not os.path.exists(log_file):
        _print_warning("No log file found.")
        _print_info("Start the Agent with 'agent start' to generate logs.")
        return 1

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        _print_error(f"Failed to read log file: {exc}")
        return 1

    n = args.lines
    recent = lines[-n:] if len(lines) > n else lines

    print(_colorize(f"=== Last {len(recent)} log lines ===", _COLOR_BOLD))
    print(f"  (from {log_file})")
    print()
    for line in recent:
        print(line.rstrip())

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构建CLI参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Self-Evolving Agent - 自驱永续进化智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  agent start                    # 使用默认配置启动
  agent start --env prod         # 使用生产环境配置启动
  agent start --config my.yaml   # 使用指定配置启动
  agent start --dry-run          # 仅验证配置不启动
  agent stop                     # 停止运行中的Agent
  agent status                   # 查询运行状态
  agent health                   # 健康检查
  agent log --lines 50           # 查看最近50行日志
  agent dashboard --port 9090    # 启动Dashboard
  agent version                  # 显示版本
""",
    )

    # --version / -V 全局标志
    parser.add_argument(
        "--version", "-V", action="store_true",
        help="显示版本信息并退出",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # start 子命令
    start_parser = subparsers.add_parser(
        "start", help="启动Agent"
    )
    start_parser.add_argument(
        "--env", default=_DEFAULT_ENV,
        help="环境 (dev/prod/test, 默认: dev)"
    )
    start_parser.add_argument(
        "--config", default=None,
        help="配置文件路径 (覆盖--env)"
    )
    start_parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别"
    )
    start_parser.add_argument(
        "--online-interval", type=float, default=None,
        help="在线循环间隔(秒)"
    )
    start_parser.add_argument(
        "--offline-interval", type=float, default=None,
        help="离线循环间隔(秒)"
    )
    start_parser.add_argument(
        "--dry-run", action="store_true",
        help="仅验证配置不启动Agent"
    )

    # stop 子命令
    stop_parser = subparsers.add_parser(
        "stop", help="停止运行中的Agent"
    )
    stop_parser.add_argument(
        "--port", type=int, default=_DEFAULT_DASHBOARD_PORT,
        help=f"Dashboard端口 (默认: {_DEFAULT_DASHBOARD_PORT})"
    )

    # status 子命令
    status_parser = subparsers.add_parser(
        "status", help="查询Agent运行状态"
    )
    status_parser.add_argument(
        "--port", type=int, default=_DEFAULT_DASHBOARD_PORT,
        help=f"Dashboard端口 (默认: {_DEFAULT_DASHBOARD_PORT})"
    )

    # health 子命令
    health_parser = subparsers.add_parser(
        "health", help="健康检查"
    )
    health_parser.add_argument(
        "--env", default=_DEFAULT_ENV,
        help="环境 (默认: dev)"
    )
    health_parser.add_argument(
        "--config", default=None,
        help="配置文件路径"
    )

    # dashboard 子命令
    dash_parser = subparsers.add_parser(
        "dashboard", help="启动独立Dashboard服务"
    )
    dash_parser.add_argument(
        "--port", type=int, default=_DEFAULT_DASHBOARD_PORT,
        help=f"端口 (默认: {_DEFAULT_DASHBOARD_PORT})"
    )

    # version 子命令
    subparsers.add_parser(
        "version", help="显示版本信息"
    )

    # log 子命令
    log_parser = subparsers.add_parser(
        "log", help="查看最近日志"
    )
    log_parser.add_argument(
        "--lines", type=int, default=20,
        help="显示行数 (默认: 20)"
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI主入口。

    Args:
        argv: 命令行参数（None时使用sys.argv）。

    Returns:
        退出码（0成功，非0失败）。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --version 全局标志
    if args.version:
        return asyncio.run(_cmd_version(args))

    if args.command is None:
        parser.print_help()
        return 0

    # 分发到子命令
    handlers = {
        "start": _cmd_start,
        "stop": _cmd_stop,
        "status": _cmd_status,
        "health": _cmd_health,
        "dashboard": _cmd_dashboard,
        "version": _cmd_version,
        "log": _cmd_log,
    }

    handler = handlers.get(args.command)
    if handler is None:
        _print_error(f"Unknown command: {args.command}")
        return 1

    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        _print_info("\nInterrupted")
        return 130
    except Exception as exc:
        _print_error(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
